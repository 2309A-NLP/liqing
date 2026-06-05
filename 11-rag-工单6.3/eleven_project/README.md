# Embedding 模型微调 — 银行-保险领域

基于 FlagEmbedding 框架，对 BAAI/bge-base-zh-v1.5 进行领域微调，提升银行-保险领域 RAG 检索效果。

## 目录结构

```
工单11/
├── scripts/
│   ├── prepare_data.py         # 数据准备：Milvus导出 → LLM生成query → BM25挖负例
│   ├── run_finetune.sh         # 模型微调（单卡 RTX 4060）
│   ├── evaluate.py             # 评估脚本（Recall@10 / NDCG@10）
│   └── run_eval_comparison.sh  # 对比基座 vs 微调模型
├── data/                       # 训练和评估数据（prepare_data.py 生成）
│   ├── train.jsonl             # 训练集
│   ├── test.jsonl              # 测试集
│   ├── test_corpus.jsonl       # 评估用语料库
│   └── test_queries.jsonl      # 评估用 query + ground truth
├── output/                     # 微调后的模型权重
└── models/                     # 本地模型（符号链接或软链接）
```

## 环境准备

```bash
# 安装依赖
pip install FlagEmbedding torch transformers datasets accelerate rank-bm25 jieba tqdm requests
```

## 使用步骤

### 1. 数据准备

从 RAG 项目的 Milvus 中导出 chunk，用 LLM 生成 query，BM25 挖负例：

```bash
export DEEPSEEK_API_KEY='your-key'
python scripts/prepare_data.py
```

产出：
- `data/train.jsonl` — 训练数据
- `data/test.jsonl` — 测试数据
- `data/test_corpus.jsonl` — 评估语料库
- `data/test_queries.jsonl` — 评估 query

### 2. 模型微调

```bash
chmod +x scripts/run_finetune.sh
./scripts/run_finetune.sh
```

训练完成后模型保存在 `output/bge-base-zh-v1.5-finetuned/`

### 3. 评估对比

```bash
chmod +x scripts/run_eval_comparison.sh
./scripts/run_eval_comparison.sh
```

自动对比基座模型和微调后的 Recall@10 / NDCG@10。

## 配置说明

| 参数 | 值 | 说明 |
|------|-----|------|
| 基座模型 | bge-base-zh-v1.5 | 768维，中文专用 |
| GPU | RTX 4060 8GB | 单卡训练 |
| batch_size | 8 | 梯度累积2步，等效16 |
| epochs | 3 | 数据量足够时3轮 |
| learning_rate | 2e-5 | 标准微调学习率 |
| max_len | 256 | query/passage 最大长度 |

## 验收标准

微调后 Recall@10 和 NDCG@10 必须优于基座模型。
