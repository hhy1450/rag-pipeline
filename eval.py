"""
RAG Pipeline 评估模块

评估的是"检索质量"，不是 LLM 生成质量。
因为检索是 RAG 的根基——检索不到正确片段，LLM 再强也没用。

指标：
  - Recall@k: top-k 个片段中，至少有一个包含正确答案的比例
  - MRR (Mean Reciprocal Rank): 第一个相关片段的排名的倒数，取平均
  - Precision@k: top-k 个片段中，有多少比例是相关的

用法：
  1. 先跑 pipeline.ingest() 索引好文档
  2. 准备 QA 对（问题 + 答案中应该出现的关键词）
  3. 调用 evaluate()，得到一个包含所有指标的结果字典

你也可以用 compare_strategies() 自动对比不同的切片策略和参数组合。
"""

import numpy as np
from itertools import product

from rag_pipeline import (
    Embedder, Generator, RAGPipeline
)


# ═══════════════════════════════════════════════════
#  QA 数据集
# ═══════════════════════════════════════════════════

# 格式: 每个条目包含问题 + 答案关键词
# 关键词是判断"检索到的片段是否相关"的依据
# 如果检索到的 chunk 中包含至少一个关键词，就认为这个 chunk 是相关的

SAMPLE_QA = [
    {
        "question": "黄海亦就读于哪所大学？",
        "keywords": ["合肥工业大学", "211"]
    },
    {
        "question": "黄海亦的GPA是多少？",
        "keywords": ["3.63", "GPA"]
    },
    {
        "question": "黄海亦的专业是什么？",
        "keywords": ["智能科学与技术"]
    },
    {
        "question": "黄海亦用过哪些编程语言？",
        "keywords": ["Python", "TypeScript", "Node.js"]
    },
    {
        "question": "OpenClaw Finance Tools 支持多少种加密货币？",
        "keywords": ["14", "加密货币"]
    },
    {
        "question": "LangBot 项目的技术栈包括哪些？",
        "keywords": ["Python", "DeepSeek", "MCP", "RAG"]
    },
    {
        "question": "黄海亦在 OpenClaw 项目中对比了哪些数据源？",
        "keywords": ["OKX", "Binance", "CoinGecko", "CryptoCompare"]
    },
    {
        "question": "黄海亦的英语水平如何？",
        "keywords": ["CET-4", "英语"]
    },
    {
        "question": "黄海亦写过的单元测试有多少个？",
        "keywords": ["11", "单元测试"]
    },
    {
        "question": "黄海亦的实习时长是多少？",
        "keywords": ["3", "4", "实习"]
    },
]


# ═══════════════════════════════════════════════════
#  核心评估逻辑
# ═══════════════════════════════════════════════════

def is_relevant(chunk: str, keywords: list[str]) -> bool:
    """判断一个 chunk 是否包含至少一个关键词。"""
    return any(kw.lower() in chunk.lower() for kw in keywords)


def evaluate(pipeline: RAGPipeline, qa_pairs: list[dict],
             k_values: list[int] = None) -> dict:
    """评估检索质量。

    Args:
        pipeline: 已经完成 ingest() 的 RAGPipeline
        qa_pairs: QA 对列表
        k_values: 要计算的 k 值列表，默认 [1, 3, 5]

    Returns:
        包含所有指标的 dict
    """
    if k_values is None:
        k_values = [1, 3, 5]
    max_k = max(k_values)

    per_question_results = []
    mrr_sum = 0.0
    recall_hits = {k: 0 for k in k_values}
    precision_hits = {k: 0.0 for k in k_values}

    for qa in qa_pairs:
        question = qa["question"]
        keywords = qa["keywords"]

        # 检索
        retrieved = pipeline.retriever.retrieve(question)
        chunks = [r["chunk"] for r in retrieved[:max_k]]
        scores = [r["score"] for r in retrieved[:max_k]]

        # 每个 chunk 是否相关
        relevance = [is_relevant(c, keywords) for c in chunks]

        # ── Recall@k: top-k 中至少有一个相关？ ──
        for k in k_values:
            if any(relevance[:k]):
                recall_hits[k] += 1

        # ── MRR: 第一个相关 chunk 的排名 ──
        first_rel_rank = None
        for rank, rel in enumerate(relevance, start=1):
            if rel:
                first_rel_rank = rank
                break
        if first_rel_rank is not None:
            mrr_sum += 1.0 / first_rel_rank

        # ── Precision@k ──
        for k in k_values:
            precision_hits[k] += sum(relevance[:k]) / k

        per_question_results.append({
            "question": question,
            "keywords": keywords,
            "chunks": chunks[:max_k],
            "scores": scores[:max_k],
            "relevance": relevance,
            "first_relevant_rank": first_rel_rank
        })

    n = len(qa_pairs)

    # 汇总指标
    metrics = {
        "total_questions": n,
        "mrr": mrr_sum / n,
    }
    for k in k_values:
        metrics[f"recall@{k}"] = recall_hits[k] / n
    for k in k_values:
        metrics[f"precision@{k}"] = precision_hits[k] / n

    return {
        "metrics": metrics,
        "details": per_question_results
    }


# ═══════════════════════════════════════════════════
#  策略对比
# ═══════════════════════════════════════════════════

def compare_strategies(pdf_path: str, embedder: Embedder, generator: Generator,
                       qa_pairs: list[dict]) -> list[dict]:
    """对比不同切片策略和 chunk_size 组合的检索效果。

    测试的组合:
      - strategy: fixed, paragraph
      - chunk_size: 300, 500, 800
      - 共 6 组实验

    Returns:
        按 recall@5 降序排列的结果列表
    """
    configs = list(product(
        ["fixed", "paragraph"],
        [300, 500, 800]
    ))

    results = []

    for strategy, chunk_size in configs:
        print(f"\n{'='*50}")
        print(f"测试: strategy={strategy}, chunk_size={chunk_size}")
        print(f"{'='*50}")

        pipeline = RAGPipeline(
            embedder=embedder,
            generator=generator,
            chunk_size=chunk_size,
            chunk_overlap=max(50, chunk_size // 5),
            split_strategy=strategy,
            top_k=5
        )
        pipeline.ingest(pdf_path)

        eval_result = evaluate(pipeline, qa_pairs, k_values=[1, 3, 5])
        metrics = eval_result["metrics"]

        results.append({
            "strategy": strategy,
            "chunk_size": chunk_size,
            "chunk_count": pipeline.store.count,
            **metrics
        })

        print(f"  chunks: {pipeline.store.count}")
        print(f"  recall@1: {metrics['recall@1']:.2%}")
        print(f"  recall@3: {metrics['recall@3']:.2%}")
        print(f"  recall@5: {metrics['recall@5']:.2%}")
        print(f"  mrr:      {metrics['mrr']:.3f}")

    # 按 recall@5 排序
    results.sort(key=lambda r: r["recall@5"], reverse=True)

    return results


# ═══════════════════════════════════════════════════
#  打印报告
# ═══════════════════════════════════════════════════

def print_report(eval_result: dict):
    """打印单次评估的详细报告。"""
    m = eval_result["metrics"]
    details = eval_result["details"]

    print(f"\n{'='*60}")
    print("评估报告")
    print(f"{'='*60}")

    print(f"\n── 总体指标 ──")
    print(f"  题目数量:    {m['total_questions']}")
    print(f"  MRR:         {m['mrr']:.4f}")
    for key in sorted(m):
        if key.startswith("recall"):
            print(f"  {key}:        {m[key]:.2%}")
        elif key.startswith("precision"):
            print(f"  {key}:     {m[key]:.2%}")

    print(f"\n── 逐题详情 ──")
    for i, d in enumerate(details):
        found = "✓" if d["first_relevant_rank"] else "✗"
        rank_str = f"第 {d['first_relevant_rank']} 位" if d["first_relevant_rank"] else "未命中"
        print(f"  [{found}] Q{i+1}: {d['question']}")
        print(f"       相关片段排名: {rank_str}")
        print(f"       Top-3 分数: {[f'{s:.3f}' for s in d['scores'][:3]]}")
        # 标注每个 chunk 是否命中
        for j, (rel, score) in enumerate(zip(d["relevance"][:3], d["scores"][:3])):
            tag = "✓ 命中" if rel else "  未命中"
            print(f"       [{j+1}] {tag} (score={score:.4f})")


def print_compare_report(results: list[dict]):
    """打印策略对比报告。"""
    print(f"\n{'='*70}")
    print("策略对比报告")
    print(f"{'='*70}")

    header = f"{'策略':<12} {'chunk大小':>10} {'chunk数':>8}  {'recall@1':>10}  {'recall@3':>10}  {'recall@5':>10}  {'MRR':>8}"
    print(header)
    print("-" * 70)

    best = results[0]
    for r in results:
        marker = " ← 最优" if r is best else ""
        print(f"{r['strategy']:<12} {r['chunk_size']:>10} {r['chunk_count']:>8}  "
              f"{r['recall@1']:>10.2%}  {r['recall@3']:>10.2%}  "
              f"{r['recall@5']:>10.2%}  {r['mrr']:>8.4f}{marker}")

    print(f"\n结论: 最优配置是 {best['strategy']} 策略 + chunk_size={best['chunk_size']}")
    print(f"       recall@5 = {best['recall@5']:.2%}, MRR = {best['mrr']:.4f}")


# ═══════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════

def main():
    import os

    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        print("请设置 DEEPSEEK_API_KEY 环境变量")
        return

    pdf_path = input("PDF 路径: ").strip()
    if not pdf_path:
        print("未输入路径，使用默认测试文件")
        pdf_path = r"C:\Users\黄海亦\Desktop\黄海亦+15327991450+AI应用开发.pdf"

    mode = input("模式: [1] 单次评估  [2] 策略对比 (默认1): ").strip()

    print("\n初始化组件...")
    embedder = Embedder("BAAI/bge-small-zh-v1.5")
    generator = Generator(api_key=api_key)

    if mode == "2":
        print("\n开始策略对比...")
        results = compare_strategies(pdf_path, embedder, generator, SAMPLE_QA)
        print_compare_report(results)
    else:
        print("\n开始单次评估...")
        pipeline = RAGPipeline(
            embedder=embedder,
            generator=generator,
            chunk_size=500,
            chunk_overlap=100,
            split_strategy="paragraph",
            top_k=5
        )
        pipeline.ingest(pdf_path)

        eval_result = evaluate(pipeline, SAMPLE_QA)
        print_report(eval_result)


if __name__ == "__main__":
    main()
