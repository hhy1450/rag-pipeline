"""
Upgraded RAG Pipeline v2 — 多格式 + 混合检索 + Reranker + RAGAS

    多格式文档 → [MultiLoader] → 纯文本
               → [TextSplitter] → chunks[]
               → [Embedder]     → dense vectors[]
               → [BM25Retriever]→ sparse index
               → [VectorStore]  → 存储
               → [HybridRetriever] → dense+sparse → RRF → Reranker → top-K
               → [Generator]    → DeepSeek 生成答案

保留原始 rag_pipeline.py 中所有组件的纯净实现。
此文件仅添加升级功能。
"""
import os
import numpy as np
from pathlib import Path

# Reuse core components from original
from rag_pipeline import (
    DocumentLoader, TextSplitter, Embedder, VectorStore,
    Retriever, Generator, RAGPipeline,
)
# New components
from loaders import get_loader, load_directory
from retrieval import HybridRetriever
from eval_ragas import evaluate_ragas, print_ragas_report, SAMPLE_QA_RAGAS


class RAGPipelineV2:
    """升级版 RAG 流水线：多格式文档 + 混合检索 + Reranker。

    用法:
        pipeline = RAGPipelineV2(embedder, generator, use_hybrid=True)
        pipeline.ingest("文件.pdf")          # 支持 PDF/Word/PPT/图片
        pipeline.ingest_directory("./docs/")  # 批量导入目录
        result = pipeline.query("你的问题")
    """

    def __init__(self, embedder: Embedder, generator: Generator,
                 chunk_size: int = 500, chunk_overlap: int = 100,
                 split_strategy: str = "paragraph", top_k: int = 5,
                 use_hybrid: bool = True,
                 use_reranker: bool = True):
        self.embedder = embedder
        self.generator = generator
        self.splitter = TextSplitter(chunk_size, chunk_overlap, split_strategy)
        self.store = VectorStore()
        # Standard dense retriever (fallback)
        self.dense_retriever = Retriever(embedder, self.store, top_k)
        # Hybrid retriever (upgrade)
        self.hybrid_retriever = None
        if use_hybrid:
            self.hybrid_retriever = HybridRetriever(
                embedder, self.store,
                use_reranker=use_reranker,
                final_top_k=top_k,
            )
        self.top_k = top_k
        self.use_hybrid = use_hybrid
        self._all_chunks: list[str] = []  # For BM25 index

    def ingest(self, file_path: str) -> int:
        """摄入单个文件（自动识别格式）。

        支持格式: PDF / Word(.docx) / PPT(.pptx) / 图片 / Markdown / TXT

        Returns: chunk 数量
        """
        file_path_obj = Path(file_path)
        if not file_path_obj.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")

        print(f"\n[1/4] 加载文件: {file_path}")
        suffix = file_path_obj.suffix.lower()

        # Use multi-format loader
        try:
            loader = get_loader(file_path)
            text = loader.load()
        except ValueError:
            # Fallback to original PDF-only loader
            loader = DocumentLoader(file_path)
            text = loader.load()

        if not text.strip():
            raise ValueError(f"文件中没有可提取的文本: {file_path}")
        print(f"  格式: {suffix}, 提取到 {len(text)} 个字符")

        return self._ingest_text(text)

    def ingest_directory(self, directory: str) -> int:
        """批量摄入目录下所有支持格式的文件。

        Returns: 总 chunk 数量
        """
        print(f"\n[批量导入] 目录: {directory}")
        docs = load_directory(directory)
        print(f"  找到 {len(docs)} 个文件")

        all_texts = []
        for doc in docs:
            if doc["text"] and not doc["text"].startswith("[加载失败"):
                all_texts.append(f"[文件: {doc['filename']}]\n{doc['text']}")
                print(f"    {doc['filename']}: {len(doc['text'])} 字符")

        if not all_texts:
            raise ValueError("目录中没有可提取文本的文件")

        combined = "\n\n".join(all_texts)
        return self._ingest_text(combined)

    def _ingest_text(self, text: str) -> int:
        """Internal: 文本 → 切片 → 向量化 → 索引 → 存储。

        Returns: chunk 数量
        """
        print(f"\n[2/4] 切片 (策略: {self.splitter.strategy}, "
              f"chunk_size={self.splitter.chunk_size})")
        chunks = self.splitter.split(text)
        print(f"  生成 {len(chunks)} 个 chunk")

        if not chunks:
            raise ValueError("没有可用的文本片段")

        print(f"\n[3/4] 向量化并存储")
        vectors = self.embedder.embed(chunks)
        self.store.add(chunks, vectors)
        self._all_chunks.extend(chunks)

        # Build BM25 index for hybrid retrieval
        if self.hybrid_retriever:
            print(f"  构建 BM25 稀疏索引...")
            self.hybrid_retriever.index(self._all_chunks)

        print(f"  已存储 {self.store.count} 条向量")
        return len(chunks)

    def query(self, question: str) -> dict:
        """问答：混合检索 → 生成 → 返回答案和引用来源。

        Returns:
            {
                "question": str,
                "answer": str,
                "sources": [{"chunk": str, "score": float, "stage": str}, ...],
                "retrieval_method": "hybrid" | "dense"
            }
        """
        # Select retriever
        if self.hybrid_retriever and self.store.count > 0:
            sources = self.hybrid_retriever.retrieve(question, top_k=self.top_k)
            method = "hybrid"
            # Normalize to standard score field
            for s in sources:
                s["score"] = s.get("rerank_score", s.get("rrf_score", 0))
        else:
            sources = self.dense_retriever.retrieve(question)
            method = "dense"
            for s in sources:
                s["stage"] = "dense"

        answer = self.generator.generate(question, sources)

        return {
            "question": question,
            "answer": answer,
            "sources": sources,
            "retrieval_method": method,
        }

    @property
    def chunk_count(self) -> int:
        return self.store.count


# ═══════════════════════════════════════
#  Demo
# ═══════════════════════════════════════

def main():
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        print("错误: 请设置环境变量 DEEPSEEK_API_KEY")
        return

    print("=" * 55)
    print("RAG Pipeline v2 — 多格式 + 混合检索 + Reranker")
    print("=" * 55)

    embedder = Embedder("BAAI/bge-small-zh-v1.5")
    generator = Generator(api_key=api_key)

    pipeline = RAGPipelineV2(
        embedder=embedder,
        generator=generator,
        chunk_size=500,
        chunk_overlap=100,
        split_strategy="paragraph",
        top_k=5,
        use_hybrid=True,
        use_reranker=True,
    )

    # 选择文件或目录
    import_path = input("\n文件路径或目录: ").strip()
    if not import_path:
        print("未输入路径，使用默认测试文件")
        import_path = r"C:\Users\黄海亦\Desktop\黄海亦+15327991450+AI应用开发.pdf"

    path_obj = Path(import_path)
    try:
        if path_obj.is_dir():
            n = pipeline.ingest_directory(import_path)
        else:
            n = pipeline.ingest(import_path)
    except (FileNotFoundError, ValueError) as e:
        print(f"错误: {e}")
        return

    print(f"\n{'=' * 55}")
    print(f"摄入完成，共 {n} 个文本片段。现在可以提问了。")
    print(f"检索模式: {'混合检索 (BM25 + Dense + RRF + Reranker)' if pipeline.use_hybrid else '稠密检索'}")
    print(f"{'=' * 55}")

    while True:
        q = input("\n> 你的问题 (输入 q 退出, eval 评估): ").strip()
        if q.lower() == 'q':
            break
        if q.lower() == 'eval':
            print("\n运行 RAGAS 评估...")
            result = evaluate_ragas(pipeline, SAMPLE_QA_RAGAS)
            print_ragas_report(result)
            continue
        if not q:
            continue

        result = pipeline.query(q)
        print(f"\n 回答:\n{result['answer']}")
        print(f"\n--- 参考片段 ({result['retrieval_method']} 检索) ---")
        for i, src in enumerate(result['sources']):
            stage = src.get('stage', '')
            score = src.get('score', 0)
            preview = src['chunk'][:120].replace('\n', ' ')
            print(f"  [{i+1}] stage={stage} score={score:.4f} | {preview}...")

    # Offer evaluation on exit
    if input("\n退出前运行 RAGAS 评估? (y/n): ").strip().lower() == 'y':
        print("\n运行 RAGAS 评估...")
        result = evaluate_ragas(pipeline, SAMPLE_QA_RAGAS)
        print_ragas_report(result)


if __name__ == "__main__":
    main()
