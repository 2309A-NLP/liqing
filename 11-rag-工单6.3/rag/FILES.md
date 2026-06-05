# RAG 问答系统 — 文件目录说明

> 项目根目录：`D:\Desktop\rag-hermes\`
> 最后更新：2026-05-22

---

## 顶层文件

| 文件 | 说明 |
|------|------|
| `rag-project-工单.md` | 任务工单，含全部技术约束（Redis / deepseek-v4-pro / bge-m3 / 混合检索 / Fitz） |
| `招股说明书1-无水印.pdf` | 目标文档 — 武汉兴图新科电子股份有限公司招股说明书 |
| `在线RAG实现方案.md` | 前期调研文档（参考用） |

---

## 代码目录结构

```
teams/team/project/
├── src/                    ← Python 源码
│   ├── config.py           ← 全局配置（环境变量 + 路径双兼容）
│   ├── logger.py           ← 日志系统（文件轮转 + queries.jsonl）
│   │
│   ├── loader/             ← 文档加载
│   │   └── pdf_loader.py   ← Fitz (pymupdf) 解析 PDF，提取文字+表格+页码
│   │
│   ├── chunker/            ← 文档分块
│   │   └── text_splitter.py ← RecursiveCharacterTextSplitter
│   │
│   ├── embedder/           ← 向量化
│   │   └── embed.py        ← bge-m3 向量化（CUDA → OOM 自动降级 CPU，带进度条）
│   │
│   ├── store/              ← 数据存储
│   │   ├── milvus_store.py  ← Milvus CRUD（自动建表/schema 校验/重建）
│   │   └── keyword_store.py ← BM25 关键词索引（jieba 分词，pickle 持久化）
│   │
│   ├── retriever/          ← 检索 + 精排
│   │   ├── hybrid_retriever.py ← 混合检索（向量×0.7 + BM25×0.3 加权融合）
│   │   │                        └─ BM25 懒加载：首次检索时从 Milvus 自动构建
│   │   └── reranker.py     ← bge-reranker Cross-Encoder 精排 Top-3
│   │
│   ├── memory/             ← 短期记忆
│   │   └── redis_memory.py  ← Redis 对话历史管理（TTL 24h，容错降级）
│   │
│   ├── generator/          ← 答案生成
│   │   └── answer_gen.py   ← deepseek-v4-pro 带引用回答（Prompt 注入检索结果+历史）
│   │
│   └── api/                ← API 服务
│       ├── main.py         ← FastAPI 入口 + 内嵌聊天界面
│       └── schemas.py      ← 请求/响应 Pydantic 模型
│
├── tests/                  ← 单元测试
│   ├── test_loader.py      ← PDF 解析测试（5 个用例）
│   ├── test_chunker.py     ← 分块测试（4 个用例）
│   ├── test_retriever.py   ← BM25 检索测试（4 个用例）
│   └── test_generator.py   ← 答案生成测试（4 个用例）
│
├── ingest.py               ← 离线入库入口（解析→分块→向量化→Milvus）
├── run.py                  ← 在线服务入口（启动 FastAPI）
│
├── setup.py                ← pip install -e . 用（PyCharm 消红线）
├── pyproject.toml          ← 项目元信息 + 依赖声明
├── requirements.txt        ← pip 依赖清单
├── README.md               ← 快速开始指南
│
├── data/                   ← 运行时数据（自动生成）
│   ├── bm25_index.pkl      ← BM25 索引缓存（懒加载自动构建）
│   └── milvus_data/        ← Milvus 持久化数据
│
└── logs/                   ← 运行时日志（自动生成）
    ├── rag.log             ← 系统运行日志（轮转 10MB × 7 天）
    └── queries.jsonl       ← 查询日志（JSON Lines 格式）
```

---

## 关键链路说明

### 离线入库 `ingest.py`

```
PDF → Fitz解析 → RecursiveCharacter分块 → bge-m3向量化 → Milvus入库
                                            ↓
                                      进度条 tqdm
```

### 在线问答 `POST /query`

```
用户问题 → config 读取环境变量 → Redis 获取历史
       → Embedder 生成查询向量 → Milvus 向量检索 Top-20
       → BM25 关键词检索 Top-20（懒加载：首次自动构建）
       → 加权融合 Top-10（向量×0.7 + BM25×0.3）
       → bge-reranker 精排 Top-3
       → deepseek-v4-pro 生成带引用回答
       → Redis 记录本轮对话
       → log_query 写入 queries.jsonl
```

### 依赖拓扑

```
config.py ─→ loader ─→ chunker ─→ embedder ─→ milvus_store
                              ↘               ↗
                           keyword_store
                                ↓
                          hybrid_retriever
                                ↓
                            reranker
                                ↓
                      redis_memory ─→ generator
                                ↓
                            api/main.py
```

---

## 启动方式

```bash
# 离线入库（一次性）
python ingest.py                    # 自动检测 PDF
python ingest.py --rebuild          # 重建库
python ingest.py --pdf path/to.pdf  # 指定文件

# 在线服务
python run.py
# 或
python -m src.api.main
# 或
uvicorn src.api.main:app --reload
```

访问 `http://localhost:8000/` → 聊天界面
访问 `http://localhost:8000/health` → 健康检查
