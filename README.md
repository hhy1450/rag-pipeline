# RAG Pipeline —— 检索增强生成

一个纯手写的中文 RAG（Retrieval-Augmented Generation）流水线，不依赖 LangChain、LlamaIndex 等框架，每个组件都是独立类，方便理解、测试和替换。

## 架构

```
PDF  →  [DocumentLoader]  →  纯文本
     →  [TextSplitter]    →  chunks[]
     →  [Embedder]        →  vectors[]
     →  [VectorStore]     →  存储 + 检索
     →  [Retriever]       →  top-k 相似片段
     →  [Generator]       →  DeepSeek 生成答案
```

| 组件 | 说明 |
|------|------|
| DocumentLoader | PyMuPDF 提取 PDF 纯文本 |
| TextSplitter | 支持 fixed（固定长度）和 paragraph（段落合并）两种策略 |
| Embedder | BAAI/bge-small-zh-v1.5，512 维中文向量，CPU 可跑 |
| VectorStore | 纯 NumPy 内存向量库，归一化后余弦相似度 = 点积 |
| Retriever | 问题向量化 → top-k 检索 |
| Generator | 调用 DeepSeek API（兼容 OpenAI SDK）生成答案 |

## 环境

- Python >= 3.10
- 依赖：`PyMuPDF`、`sentence-transformers`、`openai`、`numpy`

## 快速开始

### 1. 创建虚拟环境并安装依赖

```bash
python -m venv .venv
.venv\Scripts\activate       # Windows
pip install -r requirements.txt
```

### 2. 设置环境变量

```bash
# Windows PowerShell
$env:DEEPSEEK_API_KEY="sk-你的key"
$env:HF_ENDPOINT="https://hf-mirror.com"    # 国内镜像，解决 HuggingFace 下载问题
```

### 3. 运行

```bash
python rag_pipeline.py       # 交互式问答
python eval.py                # 检索质量评估
```

也可以直接双击 `run.bat`（需先在文件里填好 API Key）。

## 评估指标

`eval.py` 提供三个检索质量指标：

- **Recall@k** — top-k 片段中至少命中一个正确答案的比例
- **MRR**（Mean Reciprocal Rank）— 第一个相关片段排名的倒数平均值
- **Precision@k** — top-k 片段中命中的占比

支持对比不同切片策略 × chunk_size 的组合（共 6 组实验），自动输出排名。

## 文件结构

```
rag-pipeline/
├── rag_pipeline.py    # 核心流水线（6个组件 + RAGPipeline + 交互Demo）
├── eval.py             # 评估模块
├── requirements.txt    # 依赖
├── run.bat             # 一键启动脚本
└── README.md
```
