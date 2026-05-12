# RAG 问答系统 —— 保研政策智能问答

基于 **检索增强生成（Retrieval-Augmented Generation, RAG）** 架构的领域知识问答系统，以大学保研推免政策文档为知识库，支持向量检索和混合检索两种模式，并具备完整的评测体系。

## 项目结构

```
RAG/
├── VectorBase.py        # 向量数据库核心：向量化、存储、相似度检索
├── Embeddings.py        # OpenAI Embedding 封装，文本转向量
├── LLM.py               # OpenAI LLM 封装（Qwen2.5-32B-Instruct）
├── HybridSearch.py      # BM25 + 向量混合检索（RRF 融合）
├── utils.py             # 文件读取（PDF/MD/TXT）、文本分块（滑动窗口）
├── demo.py              # 交互式单条问答 Demo
├── run_eval.py          # 批量评估主脚本（向量检索 + 混合检索双路输出）
├── gettop1.py           # 生成评估草稿（自动标注 expected_source / chunk_id）
├── top10.py             # 导出每条问题的 Top-10 检索结果
├── scores.py            # 评测指标计算（检索命中率、拒答分类）
├── check_scores.py      # 检索分数分布分析（辅助调阈值）
├── eval_logger.py       # 评测日志管理器（结构化记录、CSV 汇总、历史对比）
├── search.py            # 关键词搜索工具
├── data/                # 原始文档（PDF）
│   ├── 电院关于高水平科技竞赛获奖本科生推免加分的规定（2020级）.pdf
│   └── 中南财保研工作管理办法.pdf
├── storage/             # 持久化的文档和向量
│   ├── doecment.json    # 分块后的文档
│   └── vectors.json     # 文档向量
├── eval*.jsonl          # 评估标准集（gold）
├── pred*.jsonl          # 模型预测结果
├── requirements.txt     # Python 依赖
└── eval_logs/           # 评测日志存档（每次运行生成一个子目录）
    ├── summary.csv      # 所有 runs 的指标横向对比表
    └── YYYYMMDD_HHMMSS/ # 每次评测的详细日志
```

## 快速开始

### 1. 安装依赖

```bash
conda create -n aitest01 python=3.10 -y
conda activate aitest01
pip install -r requirements.txt
```

### 2. 构建向量数据库（首次使用）

如果 `storage/` 目录下已有 `doecment.json` 和 `vectors.json`，可跳过此步。

```python
from utils import ReadFiles
from VectorBase import VectorStore
from Embeddings import OpenAIEmbedding

docs = ReadFiles('./data').get_content(max_token_len=600, cover_content=150)
vector = VectorStore(docs)
embedding = OpenAIEmbedding()
vector.get_vector(EmbeddingModel=embedding)
vector.persist(path='storage')
```

### 3. 运行 Demo

```bash
conda run -n aitest01 python demo.py
```

## 工作流程

```
原始文档 (PDF)
    │
    ▼
utils.ReadFiles  ← 读取 PDF/MD/TXT，按 token 滑动窗口分块
    │
    ▼
Embeddings.get_embedding()  ← 每块文本转向量
    │
    ▼
VectorStore.persist()  ← 保存至 storage/
    │
    ▼
┌─────────────────────────────────────┐
│         用户提问                     │
│           │                         │
│     ┌─────┴─────┐                  │
│     │  向量检索  │  BM25 检索       │
│     │  (余弦)    │  (关键词)        │
│     └─────┬─────┘                  │
│           │                         │
│     HybridSearch (RRF 融合)         │
│           │                         │
│           ▼                         │
│     Top-K 相关文档块                │
│           │                         │
│           ▼                         │
│     LLM 生成回答 (Qwen2.5-32B)      │
│           │                         │
│           ▼                         │
│     返回答案 + 引用来源             │
└─────────────────────────────────────┘
```

## 检索模式

| 模式         | 文件              | 原理                                                                       |
| ------------ | ----------------- | -------------------------------------------------------------------------- |
| **向量检索** | `VectorBase.py`   | 余弦相似度匹配语义最接近的文档块                                           |
| **混合检索** | `HybridSearch.py` | BM25 关键词匹配 + 向量语义匹配，通过 RRF (Reciprocal Rank Fusion) 融合排序 |

### RRF 融合公式

```
RRF(d) = Σ 1/(60 + rank_i(d))
```

对 BM25 和向量检索各取 Top-30，按 RRF 重排后取 Top-K。

## 拒答机制

系统采用 **双层拒答** 保障回答质量：

1. **分数阈值拒答**：最高检索分数 < `threshold`（默认 0.65），直接返回空回答
2. **LLM 语义拒答**：生成回答中包含"抱歉"等标识词时，标记为拒答

## 批量评估

### 运行评估

```bash
conda run -n aitest01 python run_eval.py
```

`run_eval.py` 会同时跑两条检索路径：

- `pred1.jsonl` —— 纯向量检索结果
- `pred_hybrid.jsonl` —— 混合检索结果

### 计算指标

修改 `scores.py` 中的 `pred_path` 指向你要评测的预测文件，然后：

```bash
conda run -n aitest01 python scores.py
```

### 评测指标

| 类别           | 指标                    | 说明                         |
| -------------- | ----------------------- | ---------------------------- |
| **检索命中率** | `hit_source@1`          | Top-1 的 source 命中率       |
|                | `hit_source@K`          | Top-K 中任意 source 命中率   |
|                | `hit_chunk@1`           | Top-1 的 chunk_id 命中率     |
|                | `hit_chunk@K`           | Top-K 中任意 chunk_id 命中率 |
| **拒答分类**   | Precision / Recall / F1 | 以 should_refuse 为正类      |
|                | TP/FP/FN/TN             | 四分类混淆矩阵               |

### 分数分布分析

用于辅助调整拒答阈值：

```bash
conda run -n aitest01 python check_scores.py
```

## 评测日志

`scores.py` 使用 `eval_logger.py` 自动记录每次评测：

```
eval_logs/
├── summary.csv              # 所有 runs 的指标横向对比
├── 20260512_093000/
│   ├── config.json          # 本次评测的配置参数
│   ├── metrics.json         # 评测指标
│   ├── predictions.jsonl    # 预测结果副本
│   └── badcases.jsonl       # 错误样本详情
└── 20260512_100000/
    └── ...
```

查看历史 runs 对比：

```python
from eval_logger import EvalLogger
EvalLogger.print_comparison()
```

## 评估数据标注流程

1. 从 `eval.jsonl`（只有问题）出发
2. 运行 `gettop1.py` → 生成 `eval_draft1.jsonl`（自动填充 source / chunk_id）
3. 人工核验 `expected_source` 和 `expected_chunk_id`，修正错误标注
4. 运行 `run_eval.py` → `scores.py` 得到指标

## 主要依赖

```
openai (>=1.0)          # OpenAI API 调用
numpy                    # 数值计算
PyPDF2                   # PDF 读取
markdown + beautifulsoup4  # Markdown 解析
tiktoken                 # Token 计数
tqdm                     # 进度条
rank_bm25                # BM25 检索
scikit-learn             # 数学工具
```

## 技术要点

- **文本分块**：基于 tiktoken (`cl100k_base`) 进行 token 级精确分块，块间有 150 token 的覆盖（cover_content），帮助模型保留上下文连贯性
- **余弦相似度**：向量检索使用余弦相似度匹配，分数范围 [-1, 1]
- **RRF 融合**：`k=60` 的 RRF 常数，对排名做倒数加权，消除两种检索量纲差异
- **OpenAI 兼容 API**：通过配置 `base_url` 和 `api_key` 支持任意兼容 OpenAI 接口的后端（vLLM 等）
- **不可回答判断**：通过 `answerable` 字段和拒答逻辑区分"知识库确实不包含"与"检索失败"

## License

Educational project.
