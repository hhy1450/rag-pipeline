"""
Minimal RAG Pipeline — from scratch, no framework magic.

    你的 PDF  →  [DocumentLoader]  →  纯文本
             →  [TextSplitter]    →  chunks[]
             →  [Embedder]        →  vectors[]
             →  [VectorStore]     →  存储 + 检索
             →  [Retriever]       →  top-k 相似片段
             →  [Generator]       →  DeepSeek 生成答案

每个组件都是一个独立的类，你可以单独测试、替换或改进任何一个环节。
"""

import os
import numpy as np
from pathlib import Path

import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
from openai import OpenAI


# ═══════════════════════════════════════════════════
#  Step 1: 文档加载器 — 把 PDF 变成纯文本
# ═══════════════════════════════════════════════════

class DocumentLoader:
    """加载 PDF 文件，提取纯文本。"""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def load(self) -> str:
        """提取单个 PDF 的全部文本。"""
        doc = fitz.open(str(self.path))
        parts = []
        for page in doc:
            text = page.get_text()
            if text.strip():
                parts.append(text)
        doc.close()
        return "\n\n".join(parts)

    @staticmethod
    def load_directory(directory: str | Path) -> list[dict]:
        """加载目录下所有 PDF，返回 [{"filename": "xxx.pdf", "text": "..."}]。"""
        docs = []
        for pdf_path in Path(directory).glob("*.pdf"):
            loader = DocumentLoader(pdf_path)
            docs.append({
                "filename": pdf_path.name,
                "text": loader.load()
            })
        return docs


# ═══════════════════════════════════════════════════
#  Step 2: 文本切片器 — 把长文本切成小块
# ═══════════════════════════════════════════════════

class TextSplitter:
    """文本切片，支持两种策略。

    fixed:      按固定字符数切分，相邻 chunk 有 overlap
    paragraph:  按自然段落切分，短的合并，避免把一个完整意思切断

    两种策略的效果不同，后面可以用评估模块对比。
    """

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100,
                 strategy: str = "paragraph"):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.strategy = strategy

    def split(self, text: str) -> list[str]:
        if self.strategy == "fixed":
            return self._fixed_split(text)
        elif self.strategy == "paragraph":
            return self._paragraph_split(text)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    def _fixed_split(self, text: str) -> list[str]:
        chunks = []
        start = 0
        step = self.chunk_size - self.chunk_overlap
        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            start += step
        return chunks

    def _paragraph_split(self, text: str) -> list[str]:
        """按双换行切段落，短段落合并直到接近 chunk_size。"""
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        if not paragraphs:
            return []

        chunks = []
        buf = paragraphs[0]

        for para in paragraphs[1:]:
            if len(buf) + len(para) < self.chunk_size:
                buf += "\n\n" + para
            else:
                chunks.append(buf)
                buf = para
        chunks.append(buf)
        return chunks


# ═══════════════════════════════════════════════════
#  Step 3: 向量化器 — 文本 → 向量
# ═══════════════════════════════════════════════════

class Embedder:
    """封装 sentence-transformers，把文本转成归一化向量。

    BGE-small-zh-v1.5: BAAI 的中文 embedding 模型，512 维。
    选它的原因：轻量（~100MB）、中文效果好、不需要 GPU 也能跑。
    """

    def __init__(self, model_name: str = "BAAI/bge-small-zh-v1.5"):
        print(f"  加载模型 {model_name} ...")
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()
        print(f"  模型加载完成，向量维度: {self.dim}")

    def embed(self, texts: str | list[str]) -> np.ndarray:
        """文本 → 归一化向量。

        Args:
            texts: 单个字符串或字符串列表

        Returns:
            单个字符串 → (dim,)
            列表 → (n, dim)
        """
        single = isinstance(texts, str)
        if single:
            texts = [texts]
        vectors = self.model.encode(texts, normalize_embeddings=True)
        return vectors[0] if single else vectors


# ═══════════════════════════════════════════════════
#  Step 4: 向量存储 — 存 + 搜
# ═══════════════════════════════════════════════════

class VectorStore:
    """内存向量库，纯 numpy 实现余弦相似度检索。

    不做索引结构（无 FAISS/ANN），暴力计算 dot product。
    数据量 < 10000 条时完全够用，且你清楚每一步在算什么。
    面试时能讲出它和 FAISS 的 trade-off 就是加分项。
    """

    def __init__(self):
        self.chunks: list[str] = []
        self.vectors: np.ndarray | None = None

    @property
    def count(self) -> int:
        return len(self.chunks)

    def add(self, chunks: list[str], vectors: np.ndarray):
        """批量添加 chunks 和对应的向量。"""
        self.chunks.extend(chunks)
        if self.vectors is None:
            self.vectors = vectors
        else:
            self.vectors = np.vstack([self.vectors, vectors])

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> list[dict]:
        """余弦相似度检索。

        因为 embedding 已归一化，cosine similarity = dot product。
        self.vectors @ query_vector 一次矩阵乘法算出所有相似度。
        """
        if self.vectors is None:
            return []

        scores = self.vectors @ query_vector  # (n,)
        top_indices = np.argsort(scores)[::-1][:top_k]

        return [
            {"chunk": self.chunks[i], "score": float(scores[i])}
            for i in top_indices
        ]


# ═══════════════════════════════════════════════════
#  Step 5: 检索器 — 问题 → 相关片段
# ═══════════════════════════════════════════════════

class Retriever:
    """接收用户问题，返回最相关的 top-k 文档片段。

    它做的事：
    1. 用同一个 Embedder 把问题向量化
    2. 在 VectorStore 里搜 top-k 相似片段
    3. 返回 chunks + 相似度分数
    """

    def __init__(self, embedder: Embedder, vector_store: VectorStore, top_k: int = 5):
        self.embedder = embedder
        self.store = vector_store
        self.top_k = top_k

    def retrieve(self, query: str) -> list[dict]:
        query_vec = self.embedder.embed(query)
        return self.store.search(query_vec, top_k=self.top_k)


# ═══════════════════════════════════════════════════
#  Step 6: 生成器 — 片段 + 问题 → LLM → 答案
# ═══════════════════════════════════════════════════

class Generator:
    """调用 DeepSeek API 生成最终答案。

    DeepSeek 兼容 OpenAI SDK，只需改 base_url。
    """

    SYSTEM_PROMPT = """你是一个知识助手。你需要根据提供的文档片段回答用户的问题。

规则：
1. 只使用文档片段中提供的信息来回答
2. 如果文档中没有相关信息，直接说"文档中没有提到相关内容"
3. 回答要简洁、准确，不要编造信息
4. 引用文档内容时，用引号标注原文"""

    def __init__(self, api_key: str,
                 base_url: str = "https://api.deepseek.com",
                 model: str = "deepseek-chat"):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def generate(self, query: str, retrieved_chunks: list[dict]) -> str:
        """构造 prompt，发送给 DeepSeek，返回答案。"""
        context = "\n\n---\n\n".join(
            f"[片段 {i+1}] {c['chunk']}"
            for i, c in enumerate(retrieved_chunks)
        )

        user_prompt = f"""以下是检索到的文档片段：

{context}

用户问题：{query}

请根据以上文档片段回答用户的问题。"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            max_tokens=800
        )
        return response.choices[0].message.content


# ═══════════════════════════════════════════════════
#  Step 7: 主控 — 把以上 6 个组件串起来
# ═══════════════════════════════════════════════════

class RAGPipeline:
    """完整的 RAG 流水线。

    用法：
        pipeline = RAGPipeline(embedder, generator, chunk_size=500, top_k=5)
        pipeline.ingest("你的文件.pdf")
        result = pipeline.query("你的问题")
    """

    def __init__(self, embedder: Embedder, generator: Generator,
                 chunk_size: int = 500, chunk_overlap: int = 100,
                 split_strategy: str = "paragraph", top_k: int = 5):
        self.embedder = embedder
        self.generator = generator
        self.splitter = TextSplitter(chunk_size, chunk_overlap, split_strategy)
        self.store = VectorStore()
        self.retriever = Retriever(embedder, self.store, top_k)

    def ingest(self, pdf_path: str) -> int:
        """摄入一个 PDF：加载 → 切片 → 向量化 → 存库。返回 chunk 数量。"""
        if not Path(pdf_path).exists():
            raise FileNotFoundError(f"文件不存在: {pdf_path}")

        print(f"\n[1/3] 加载 PDF: {pdf_path}")
        loader = DocumentLoader(pdf_path)
        text = loader.load()
        if not text.strip():
            raise ValueError("PDF 中没有可提取的文本（可能是扫描件）")
        print(f"  提取到 {len(text)} 个字符")

        print(f"\n[2/3] 切片 (策略: {self.splitter.strategy}, "
              f"chunk_size={self.splitter.chunk_size})")
        chunks = self.splitter.split(text)
        print(f"  生成 {len(chunks)} 个 chunk")

        print(f"\n[3/3] 向量化并存储")
        vectors = self.embedder.embed(chunks)
        self.store.add(chunks, vectors)
        print(f"  已存储 {self.store.count} 条向量")

        return len(chunks)

    def query(self, question: str) -> dict:
        """问答：检索 → 生成 → 返回答案和引用来源。"""
        results = self.retriever.retrieve(question)
        answer = self.generator.generate(question, results)
        return {
            "question": question,
            "answer": answer,
            "sources": results
        }


# ═══════════════════════════════════════════════════
#  交互式 Demo
# ═══════════════════════════════════════════════════

def main():
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        print("错误: 请设置环境变量 DEEPSEEK_API_KEY")
        print('  在终端执行: export DEEPSEEK_API_KEY="sk-your-key"')
        return

    # 1. 初始化组件
    print("=" * 50)
    print("初始化 RAG Pipeline")
    print("=" * 50)

    embedder = Embedder("BAAI/bge-small-zh-v1.5")
    generator = Generator(api_key=api_key)

    pipeline = RAGPipeline(
        embedder=embedder,
        generator=generator,
        chunk_size=500,
        chunk_overlap=100,
        split_strategy="paragraph",
        top_k=5
    )

    # 2. 摄入 PDF
    pdf_path = input("\n请输入 PDF 文件路径: ").strip()
    try:
        n = pipeline.ingest(pdf_path)
    except (FileNotFoundError, ValueError) as e:
        print(f"错误: {e}")
        return

    print(f"\n{'=' * 50}")
    print(f"摄入完成，共 {n} 个文本片段。现在可以提问了。")
    print(f"{'=' * 50}")

    # 3. 交互式问答
    while True:
        q = input("\n> 你的问题 (输入 q 退出): ").strip()
        if q.lower() == 'q':
            break
        if not q:
            continue

        result = pipeline.query(q)
        print(f"\n📖 回答:\n{result['answer']}")
        print(f"\n--- 参考片段 ---")
        for i, src in enumerate(result['sources']):
            preview = src['chunk'][:120].replace('\n', ' ')
            print(f"  [{i+1}] score={src['score']:.4f} | {preview}...")


if __name__ == "__main__":
    main()
