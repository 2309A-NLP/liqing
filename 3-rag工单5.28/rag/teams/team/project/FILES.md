# FILES.md — 项目文件说明

RAG 问答系统（基于 PDF 文档），工单编号：人工智能NLP-RAG-基于PDF文档的问答系统

## 目录结构

```
project/
├── run.py                    # API 启动入口
├── ingest.py                 # 离线入库脚本（MinerU → 分块 → 向量化 → Milvus）
├── pyproject.toml            # 项目配置 + 依赖声明
├── requirements.txt          # pip install -r 依赖清单
├── README.md                 # 项目说明
├── .gitignore                # Git 忽略规则
│
├── src/                      # 源码包
│   ├── __init__.py
│   ├── config.py             # 全局配置（环境变量优先，Windows/WSL 双兼容）
│   ├── logger.py             # 日志模块（结构化查询日志 + 文件轮转）
│   │
│   ├── api/                  # FastAPI 在线服务
│   │   ├── __init__.py
│   │   ├── main.py           # 应用入口，/ask /upload /health 路由
│   │   └── schemas.py        # Pydantic 数据模型（QueryRequest / QueryResponse 等）
│   │
│   ├── loader/               # 文档加载
│   │   ├── __init__.py
│   │   └── mineru_loader.py  # MinerU content_list.json 加载器（噪声过滤 + 短块合并）
│   │
│   ├── chunker/              # 文档分块
│   │   ├── __init__.py
│   │   └── text_splitter.py  # 分块器：text 按 RecursiveCharacterTextSplitter 切分，
│   │                         #         table 双 chunk（table_semantic + table_json），
│   │                         #         header 注入 section_path 上下文
│   │
│   ├── embedder/             # 向量化
│   │   ├── __init__.py
│   │   └── embed.py          # bge-m3 embedding（sentence-transformers）
│   │
│   ├── store/                # 存储层
│   │   ├── __init__.py
│   │   ├── milvus_store.py   # Milvus 向量库（insert / search / delete_all / count）
│   │   └── keyword_store.py  # BM25 关键词索引（jieba 分词 + rank-bm25）
│   │
│   ├── retriever/            # 检索层
│   │   ├── __init__.py
│   │   ├── hybrid_retriever.py # 混合检索：向量 + BM25 加权融合
│   │   └── reranker.py       # bge-reranker 精排（GPU 加速）
│   │
│   ├── query/                # 查询理解
│   │   ├── __init__.py
│   │   └── understander.py   # LLM 驱动：指代消解、问题分解、意图提取
│   │
│   ├── generator/            # 答案生成
│   │   ├── __init__.py
│   │   └── answer_gen.py     # DeepSeek 答案生成（结构化 prompt + 来源引用）
│   │
│   └── memory/               # 对话记忆
│       ├── __init__.py
│       └── redis_memory.py   # Redis 短期记忆（对话历史存取）
│
├── tests/                    # 测试 + 诊断 + 评测
│   ├── test_all.py           # 全量测试 runner（详细报告模式）
│   ├── test_chunker.py       # 分块模块单元测试
│   ├── test_generator.py     # 答案生成模块单元测试
│   ├── test_retriever.py     # 检索模块单元测试
│   ├── test_bm25.py          # BM25 索引快速验证
│   ├── test_embed.py         # Embedder 独立诊断（排查 segfault）
│   ├── check_retrieval.py    # 检索质量检查（端到端向量+BM25）
│   └── eval_ragas.py         # RAGAS 评测脚本（faithfulness/relevancy/precision/recall）
│
├── data/                     # 运行时数据（.gitignore 屏蔽）
│   ├── bm25_index.pkl        # BM25 索引持久化
│   └── preview/              # ingest.py --preview 预览输出
│
└── logs/                     # 日志目录（.gitignore 屏蔽）
    ├── rag.log               # 应用日志
    └── queries.jsonl         # 结构化查询日志
```

## 数据流

### 离线入库（一次性）
```
PDF → MinerU 解析（离线工具）
    → data/source_docs/*_content_list.json
    → ingest.py
        → MinerULoader（噪声过滤 + 短块合并）
        → Chunker（text/table 双策略分块 + section_path）
        → Embedder（bge-m3 向量化）
        → Milvus 入库 + BM25 索引
```

### 在线查询
```
用户问题 → QueryUnderstander（指代消解 + 问题分解）
         → HybridRetriever（向量 + BM25 融合检索）
         → Reranker（bge-reranker 精排）
         → AnswerGenerator（DeepSeek 生成答案 + 来源引用）
         → Redis 记忆更新
         → 返回答案
```

## 关键配置（config.py）

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| MILVUS_URI | http://localhost:19530 | Milvus 地址 |
| COLLECTION_NAME | rag_chunks | 集合名 |
| EMBED_MODEL_PATH | bge-m3 模型路径 | Embedding 模型 |
| CHUNK_SIZE | 512 | 分块最大 token |
| CHUNK_OVERLAP | 64 | 分块重叠 |
| MIN_CHUNK_LEN | 30 | 短 chunk 过滤阈值 |
| RERANK_TOP_N | 5 | Reranker 精排候选数 |
| API_PORT | 8004 | API 端口 |
| DEEPSEEK_API_KEY | - | DeepSeek API 密钥 |

## 依赖

- pymilvus >= 2.4.0（向量库）
- sentence-transformers >= 3.0.0（bge-m3 embedding + bge-reranker）
- fastapi + uvicorn（在线 API）
- rank-bm25 + jieba（BM25 关键词检索）
- redis（对话记忆）
- langchain-text-splitters（文本切分）
- ragas（评测，仅 tests/eval_ragas.py 需要）
