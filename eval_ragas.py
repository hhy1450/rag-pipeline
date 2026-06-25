"""
RAGAS 评估模块 —— 四维量化评估 RAG 系统效果。

指标:
  - Faithfulness (答案忠实度):    生成答案是否忠实于检索到的上下文
  - Answer Relevancy (答案相关性): 答案与问题的相关程度
  - Context Precision (上下文精确度): 检索到的上下文中相关信息的占比
  - Context Recall (上下文召回率):   答案所需信息是否在检索上下文中

用法:
    from eval_ragas import evaluate_ragas, print_ragas_report

    result = evaluate_ragas(pipeline, qa_pairs)
    print_ragas_report(result)
"""
import numpy as np
from datasets import Dataset


def evaluate_ragas(pipeline, qa_pairs: list[dict]) -> dict:
    """使用 RAGAS 库评估 RAG Pipeline。

    Args:
        pipeline: 已完成 ingest() 的 RAGPipeline（需有 query 方法返回 sources）
        qa_pairs: QA 对列表，每项含 {"question": str, "ground_truth": str}

    Returns:
        {"metrics": {...}, "details": [...]} 评估结果
    """
    try:
        from ragas import evaluate
        from ragas.metrics import (
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        )
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import HuggingFaceEmbeddings
        from langchain_openai import ChatOpenAI
    except ImportError as e:
        print(f"RAGAS 依赖缺失: {e}")
        print("安装: pip install ragas langchain-openai")
        return _fallback_eval(pipeline, qa_pairs)

    import os

    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        print("请设置 DEEPSEEK_API_KEY 环境变量用于 RAGAS 评估")
        return _fallback_eval(pipeline, qa_pairs)

    # RAGAS 需要 LLM 和 Embeddings 来做评估
    eval_llm = LangchainLLMWrapper(ChatOpenAI(
        model="deepseek-chat",
        api_key=api_key,
        base_url="https://api.deepseek.com",
        temperature=0.0,
    ))
    eval_embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-zh-v1.5",
    )

    # 构建 RAGAS 数据集
    questions = []
    answers = []
    contexts = []
    ground_truths = []

    for qa in qa_pairs:
        result = pipeline.query(qa["question"])
        questions.append(qa["question"])
        answers.append(result["answer"])
        contexts.append([c["chunk"] for c in result["sources"]])
        ground_truths.append(qa.get("ground_truth", ""))

    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    })

    # 执行 RAGAS 评估
    print("\n运行 RAGAS 评估 (4 个指标)...")
    metrics_list = [faithfulness, answer_relevancy, context_precision, context_recall]
    result = evaluate(
        dataset=dataset,
        metrics=metrics_list,
        llm=eval_llm,
        embeddings=eval_embeddings,
    )

    # Extract scores
    scores = {}
    if result.get("faithfulness") is not None:
        scores["faithfulness"] = float(np.mean([s for s in result["faithfulness"] if s is not None]))
    if result.get("answer_relevancy") is not None:
        scores["answer_relevancy"] = float(np.mean([s for s in result["answer_relevancy"] if s is not None]))
    if result.get("context_precision") is not None:
        scores["context_precision"] = float(np.mean([s for s in result["context_precision"] if s is not None]))
    if result.get("context_recall") is not None:
        scores["context_recall"] = float(np.mean([s for s in result["context_recall"] if s is not None]))

    return {
        "metrics": scores,
        "details": [
            {"question": q, "answer": a, "contexts": c, "ground_truth": g}
            for q, a, c, g in zip(questions, answers, contexts, ground_truths)
        ],
    }


def _fallback_eval(pipeline, qa_pairs: list[dict]) -> dict:
    """Fallback evaluation when RAGAS is not available.

    Uses simple keyword matching as a minimal check.
    """
    print("\n使用关键词匹配进行基础评估...")

    total = len(qa_pairs)
    relevant = 0
    details = []

    for qa in qa_pairs:
        question = qa["question"]
        result = pipeline.query(question)
        answer = result["answer"]
        sources = result["sources"]

        # Check if answer contains expected keywords
        keywords = qa.get("keywords", [])
        matched = any(kw.lower() in answer.lower() for kw in keywords)

        details.append({
            "question": question,
            "answer": answer[:200],
            "contexts": [c["chunk"][:100] for c in sources[:3]],
            "keywords": keywords,
            "keywords_matched": matched,
        })

        if matched:
            relevant += 1

    return {
        "metrics": {
            "keyword_match_rate": relevant / total if total > 0 else 0,
            "note": "RAGAS not available. Install: pip install ragas langchain-openai",
        },
        "details": details,
    }


def print_ragas_report(result: dict):
    """打印 RAGAS 评估报告。"""
    metrics = result["metrics"]
    details = result["details"]

    print(f"\n{'='*60}")
    print("RAGAS 评估报告")
    print(f"{'='*60}")

    print(f"\n── 总体指标 ──")
    for name, value in metrics.items():
        if isinstance(value, float):
            print(f"  {name:<25s}: {value:.2%}")
        else:
            print(f"  {name:<25s}: {value}")

    print(f"\n── 逐题详情 ──")
    for i, d in enumerate(details):
        print(f"  Q{i+1}: {d['question']}")
        print(f"       Answer: {d['answer'][:150]}...")
        if "contexts" in d:
            print(f"       Top context (preview): {d['contexts'][0][:80]}...")
            print(f"       Context count: {len(d['contexts'])}")
        if "keywords" in d:
            matched = "✓" if d.get("keywords_matched") else "✗"
            print(f"       Keywords matched: {matched}")
        print()


# ---- Example QA pairs (same as your original SAMPLE_QA) ----

SAMPLE_QA_RAGAS = [
    {
        "question": "黄海亦就读于哪所大学？",
        "ground_truth": "黄海亦就读于合肥工业大学。",
        "keywords": ["合肥工业大学", "211"],
    },
    {
        "question": "黄海亦的GPA是多少？",
        "ground_truth": "黄海亦的GPA是3.63。",
        "keywords": ["3.63", "GPA"],
    },
    {
        "question": "黄海亦的专业是什么？",
        "ground_truth": "黄海亦的专业是智能科学与技术。",
        "keywords": ["智能科学与技术"],
    },
    {
        "question": "黄海亦用过哪些编程语言？",
        "ground_truth": "黄海亦使用过Python、TypeScript、Node.js等编程语言。",
        "keywords": ["Python", "TypeScript", "Node.js"],
    },
    {
        "question": "OpenClaw Finance Tools 支持多少种加密货币？",
        "ground_truth": "OpenClaw Finance Tools 支持14种加密货币。",
        "keywords": ["14", "加密货币"],
    },
    {
        "question": "LangBot 项目的技术栈包括哪些？",
        "ground_truth": "LangBot项目使用了Python、DeepSeek、MCP、RAG等技术。",
        "keywords": ["Python", "DeepSeek", "MCP", "RAG"],
    },
]


def main():
    """Quick self-test."""
    import os
    from rag_pipeline import Embedder, Generator, RAGPipeline

    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        print("请设置 DEEPSEEK_API_KEY")
        return

    pdf_path = input("PDF 路径: ").strip()
    if not pdf_path:
        pdf_path = r"C:\Users\黄海亦\Desktop\黄海亦+15327991450+AI应用开发.pdf"

    # Build pipeline
    embedder = Embedder("BAAI/bge-small-zh-v1.5")
    generator = Generator(api_key=api_key)
    pipeline = RAGPipeline(embedder=embedder, generator=generator)
    pipeline.ingest(pdf_path)

    # Evaluate
    result = evaluate_ragas(pipeline, SAMPLE_QA_RAGAS)
    print_ragas_report(result)


if __name__ == "__main__":
    main()
