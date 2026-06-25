"""
Hybrid retrieval with two-stage ranking.

Stage 1 (召回): BM25 稀疏 + Dense 稠密 → RRF 双路融合 → top-N
Stage 2 (精排): BGE-Reranker 对召回结果重排序 → top-K

用法:
    retriever = HybridRetriever(embedder, vector_store)
    retriever.index(chunks)                      # 构建 BM25 索引
    results = retriever.retrieve("问题", top_k=5) # 混合检索 + 精排
"""
import numpy as np
from rank_bm25 import BM25Okapi


class BM25Retriever:
    """BM25 稀疏检索器。

    基于词频的经典检索算法，对关键词匹配非常敏感，
    与稠密语义检索互补。
    """

    def __init__(self):
        self._corpus: list[str] = []
        self._bm25: BM25Okapi | None = None

    def index(self, chunks: list[str]):
        """构建 BM25 索引。

        Args:
            chunks: 已切分的文本片段列表。
        """
        from rank_bm25 import BM25Okapi
        self._corpus = chunks
        # Tokenize: simple whitespace split (Chinese chars are individual)
        tokenized = [list(chunk) for chunk in chunks]
        self._bm25 = BM25Okapi(tokenized)

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        """BM25 检索，返回 top-k 结果。

        Returns:
            [{"chunk": str, "score": float}, ...]
            Scores are normalized to [0, 1].
        """
        if self._bm25 is None:
            return []

        tokenized = list(query)
        scores = self._bm25.get_scores(tokenized)

        # Normalize to [0, 1]
        max_score = scores.max() if len(scores) > 0 else 1.0
        if max_score == 0:
            max_score = 1.0
        normed = scores / max_score

        top_indices = np.argsort(normed)[::-1][:top_k]
        return [
            {"chunk": self._corpus[i], "score": float(normed[i])}
            for i in top_indices
        ]


def reciprocal_rank_fusion(
    dense_results: list[dict],
    sparse_results: list[dict],
    k: int = 60,
    top_k: int = 10,
) -> list[dict]:
    """RRF (Reciprocal Rank Fusion) 双路融合。

    稠密检索擅长语义匹配，稀疏检索擅长关键词匹配。
    RRF 将两者的排名信息融合，不需要分数校准。

    RRF_score(doc) = Σ 1/(k + rank_i(doc))

    Args:
        dense_results: 稠密检索结果列表
        sparse_results: 稀疏检索结果列表
        k: RRF 常数（默认 60，经典值）
        top_k: 返回 top-k 融合结果

    Returns:
        按 RRF 分数降序排列的 chunks + 分数
    """
    # Build chunk lookup for each result set
    dense_ranks = {}
    for rank, r in enumerate(dense_results):
        dense_ranks[r["chunk"][:200]] = {
            "rank": rank + 1,
            "dense_score": r["score"],
            "chunk": r["chunk"],
        }

    sparse_ranks = {}
    for rank, r in enumerate(sparse_results):
        sparse_ranks[r["chunk"][:200]] = {
            "rank": rank + 1,
            "sparse_score": r["score"],
            "chunk": r["chunk"],
        }

    # Compute RRF scores
    rrf_scores: dict[str, dict] = {}

    all_keys = set(dense_ranks.keys()) | set(sparse_ranks.keys())
    for key in all_keys:
        rrf = 0.0
        if key in dense_ranks:
            rrf += 1.0 / (k + dense_ranks[key]["rank"])
        if key in sparse_ranks:
            rrf += 1.0 / (k + sparse_ranks[key]["rank"])

        # Use chunk from either source
        chunk = dense_ranks[key]["chunk"] if key in dense_ranks else sparse_ranks[key]["chunk"]
        rrf_scores[key] = {
            "chunk": chunk,
            "rrf_score": rrf,
            "dense_score": dense_ranks[key]["dense_score"] if key in dense_ranks else 0,
            "sparse_score": sparse_ranks[key]["sparse_score"] if key in sparse_ranks else 0,
        }

    # Sort by RRF score
    sorted_items = sorted(rrf_scores.values(), key=lambda x: x["rrf_score"], reverse=True)
    return sorted_items[:top_k]


class Reranker:
    """BGE-Reranker 精排器。

    在粗排（RRF 融合）得到 top-N 候选后，用精排模型对每个
    (query, chunk) 对打分，排序更准确。

    模型: BAAI/bge-reranker-base 或 BAAI/bge-reranker-v2-m3
    """

    def __init__(self, model_name: str = "BAAI/bge-reranker-base"):
        from sentence_transformers import CrossEncoder
        print(f"  加载 Reranker: {model_name} ...")
        self.model = CrossEncoder(model_name)
        self.model_name = model_name

    def rerank(self, query: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
        """对候选片段精排。

        Args:
            query: 用户问题
            candidates: 粗排结果列表 (每个含 "chunk" 字段)
            top_k: 返回 top-k 精排结果

        Returns:
            按精排分数降序排列的结果，标记 rerank_score 和 stage 字段
        """
        if not candidates:
            return []

        # Build (query, chunk) pairs
        pairs = [(query, c["chunk"]) for c in candidates]
        scores = self.model.predict(pairs)

        # Sort by reranker score
        sorted_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in sorted_indices:
            c = candidates[idx].copy()
            c["rerank_score"] = float(scores[idx])
            c["stage"] = "reranker"
            results.append(c)
        return results


class HybridRetriever:
    """两阶段混合检索器。

    Stage 1: BM25(sparse) + Dense → RRF fusion → top-N 粗排候选
    Stage 2: BGE-Reranker → top-K 精排结果

    用法:
        retriever = HybridRetriever(embedder, vector_store)
        retriever.index(chunks)
        results = retriever.retrieve("问题")
    """

    def __init__(self, embedder, vector_store,
                 reranker_model: str = "BAAI/bge-reranker-base",
                 use_reranker: bool = True,
                 fusion_top_n: int = 20,
                 final_top_k: int = 5):
        self.embedder = embedder
        self.store = vector_store
        self.bm25 = BM25Retriever()
        self.reranker = Reranker(reranker_model) if use_reranker else None
        self.fusion_top_n = fusion_top_n
        self.final_top_k = final_top_k
        self.use_reranker = use_reranker

    def index(self, chunks: list[str]):
        """构建 BM25 稀疏索引。

        稠密索引已在 VectorStore.add() 时完成，只需额外构建 BM25。
        """
        self.bm25.index(chunks)

    def retrieve(self, query: str, top_k: int | None = None) -> list[dict]:
        """两阶段混合检索。

        Args:
            query: 用户问题
            top_k: 最终返回数量（覆盖实例默认值）

        Returns:
            精排后的检索结果列表
        """
        if top_k is None:
            top_k = self.final_top_k

        # ── Stage 1: 双路召回 + RRF 融合 ──
        # Dense retrieval
        query_vec = self.embedder.embed(query)
        dense_results = self.store.search(query_vec, top_k=self.fusion_top_n)

        # Sparse (BM25) retrieval
        sparse_results = self.bm25.search(query, top_k=self.fusion_top_n)

        # RRF fusion
        fused = reciprocal_rank_fusion(
            dense_results, sparse_results,
            top_k=self.fusion_top_n,
        )

        # ── Stage 2: Reranker 精排 ──
        if self.reranker is not None and len(fused) > 0:
            return self.reranker.rerank(query, fused, top_k=top_k)

        # Fallback: 无 reranker 时直接用 RRF 结果
        for c in fused:
            c["rerank_score"] = c["rrf_score"]
            c["stage"] = "rrf"
        return fused[:top_k]
