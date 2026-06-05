# RAG 项目 — 基于 PDF 文档的问答系统 · 任务工单

> 来源：`人工智能NLP-RAG项目-01-基于PDF文档的问答系统任务工单V1.1-20250206.pdf`
> 工单编号：`rag-project-sprint-1`
> 创建时间：2026-05-22（更新：2026-05-22）
> 工时预估：2 人日

---

## 总目标

基于大语言模型（LLM）+ RAG 技术，构建一个**针对《招股说明书.pdf》内容进行问答检索**的系统。核心链路：

```
用户提问 → Query理解 → 混合检索(向量+BM25) → Reranker → 答案生成 → Redis记忆 → 界面展示
                                                              ↑
                                                    短期对话历史存储
```

---

## 第一阶段：Product Agent（调研与规划）

### 调研内容

| 维度 | 内容 |
|------|------|
| 场景 | 内部文档问答，目标文档是招股说明书类 PDF（长文档、表格多、结构化数据） |
| 技术栈 | Python + LangChain/LlamaIndex + FastAPI + Redis |
| 分块策略 | RecursiveCharacterTextSplitter（chunk_size=512, overlap=128），需支持表格保留 |
| Embedding | **bge-m3**（本地路径 `D:\models\bge-m3`，WSL2 映射 `/mnt/d/models/bge-m3`） |
| Reranker | **bge-reranker**（本地路径 `D:\models\bge-reranker`，WSL2 映射 `/mnt/d/models/bge-reranker`） |
| LLM | **deepseek-v4-pro**（API Key 从永久环境变量 `DEEPSEEK_API_KEY` 读取） |
| 向量库 | **Milvus**（向量数据库，存储分块后的向量数据） |
| 短期记忆 | **Redis**（已部署 Docker，端口 6379，存储对话历史用于上下文连续） |
| PDF 解析 | **Fitz**（pymupdf 底层引擎） |

### Milvus 配置说明

| 配置项 | 值 |
|--------|-----|
| 类型 | 向量数据库，存储分块后的向量 + 文本 |
| 连接方式 | `pymilvus` 客户端连接 |
| 集合（Collection） | `pdf_chunks` — 字段：`id, vector, text, page_no, source_file` |
| 索引类型 | IVF_FLAT 或 HNSW（视数据量选择） |
| 部署 | 如未部署，需 `docker run -d --name rag-milvus -p 19530:19530 milvusdb/milvus:latest` |

### Redis 说明

- Redis 已在 Docker 中部署运行，端口 6379
- 每条对话写入 Redis，保留最近 N 轮上下文，支持会话 ID 隔离
- 无需重新部署

### 产出物

1. `teams/team/artifacts/prd/YYYY-MM-DD-prd.md` — 详细 PRD
2. `teams/team/artifacts/tasks.md` — 拆分后的开发任务
3. `teams/team/artifacts/handover-to-dev.md` — 交接文档

---

## 第二阶段：Dev Agent（系统实现）

### 功能需求

#### 1. Query 理解模块
- 意图识别：识别用户问题的核心意图
- 消歧：处理多义词或模糊表述
- 分解与抽象：将复杂问题分解为多个子问题，提取关键信息
- 结合历史上下文：读取 Redis 中最近对话记录，辅助理解当前问题

#### 2. 文档解析模块（Fitz）
- 使用 **Fitz（pymupdf）** 解析《招股说明书.pdf》
- 提取文字内容 + 表格数据
- 保留页码信息（用于引用溯源）
- 支持中英文混合文档
- 异常处理：损坏 PDF / 空文档 / 加密文档

#### 3. 混合检索模块（向量 + BM25）
- **向量检索**：文档分块 → bge-m3 向量化 → Milvus 检索 Top-K=20
- **BM25 关键词检索**：使用 `rank_bm25` 库，同文档库做关键词检索 Top-K=20
- **混合策略**：向量分 × 0.7 + BM25 分 × 0.3 加权融合 → 取 Top-K=10
- **Reranker**：bge-reranker 对 Top-10 精排 → 取 Top-N=3 送入 LLM

#### 4. 短期记忆模块（Redis）
- 每次问答完成后将 `(问题, 答案, 引用来源)` 写入 Redis
- Key 格式：`session:{session_id}:history:{timestamp}`
- 支持按 `session_id` 隔离多轮对话
- 支持 LLM 注入最近 N 轮上下文
- TTL 过期策略（默认 24 小时自动清理）

#### 5. 答案生成模块
- 取 Reranker 后的 Top-3 块作为上下文
- 拼接最近 N 轮对话历史（从 Redis 读取）
- 调用 deepseek-v4-pro（`DEEPSEEK_API_KEY` 环境变量）
- 生成带引用的回答（标注来源页码 + 原文段落）
- 支持流式输出（SSE，可选）

#### 6. 知识库管理模块
- 支持解析后文档的存储和管理
- 增量更新（后续支持上传新文档）

#### 7. 交互界面
- Web 界面（FastAPI + 简单前端）
- 支持问题输入和答案展示
- 显示引用来源（原文段落 + 页码）
- 多轮对话上下文展示

### 技术约束

- 代码路径：`teams/team/project/src/`
- 注释必须包含工单编号：`人工智能NLP-RAG-基于PDF文档的问答系统`
- 端到端响应时间 ≤ 3 秒（含检索 + rerank + LLM）
- Milvus 连接配置：`host=localhost, port=19530, collection=pdf_chunks`
- Redis 连接配置：`host=localhost, port=6379`（**已部署，无需启动**）
- LLM API Key 从环境变量 `DEEPSEEK_API_KEY` 读取（**不要硬编码**）

### 产出物

```
teams/team/project/
├── src/
│   ├── __init__.py
│   ├── config.py              # 全局配置（模型路径、Redis地址、API Key等）
│   ├── loader/                # 文档加载器（PDF 解析）
│   │   ├── __init__.py
│   │   └── pdf_loader.py      # 使用 Fitz (pymupdf) 解析
│   ├── chunker/               # 分块策略
│   │   ├── __init__.py
│   │   └── text_splitter.py
│   ├── embedder/              # Embedding 调用
│   │   ├── __init__.py
│   │   └── embed.py           # 调用 bge-m3
│   ├── store/                 # 向量库 CRUD
│   │   ├── __init__.py
│   │   ├── milvus_store.py    # Milvus 向量库操作（插入/检索）
│   │   └── keyword_store.py   # BM25 关键词索引
│   ├── retriever/             # 混合检索 + Rerank
│   │   ├── __init__.py
│   │   ├── hybrid_retriever.py  # 向量+BM25 加权融合
│   │   └── reranker.py        # bge-reranker 精排
│   ├── memory/                # 短期记忆（Redis）
│   │   ├── __init__.py
│   │   └── redis_memory.py    # Redis CRUD，对话历史管理
│   ├── generator/             # 回答生成
│   │   ├── __init__.py
│   │   └── answer_gen.py      # deepseek-v4-pro + 上下文注入
│   └── api/                   # FastAPI 服务
│       ├── __init__.py
│       ├── main.py
│       └── schemas.py
├── tests/
│   ├── test_loader.py
│   ├── test_chunker.py
│   ├── test_retriever.py
│   └── test_generator.py
├── requirements.txt
├── pyproject.toml
├── README.md
└── data/
    └── milvus_data/           # Milvus 数据持久化目录
```

### Redis 数据模型

```
会话数据结构：
  session:{session_id}:history:{timestamp}
    → {"role": "user" | "assistant", "content": "...", "sources": [...]}
  TTL: 86400s（24小时自动过期）
```

### 测试问题列表（用于开发阶段验证）

```
1. 报告期内，武汉兴图新科电子股份有限公司来自军用领域的收入分别是多少？
2. 武汉兴图新科电子股份有限公司参与制定了哪个技术标准？
3. 报告期内，武汉兴图新科电子股份有限公司来自军用领域的收入占主营业务收入的比重分别是多少？
4. 根据武汉兴图新科电子股份有限公司招股意向书，电子信息行业的上游涉及哪些企业？
5. 武汉兴图新科电子股份有限公司在哪个领域已经成为重要供应商？
6. 根据武汉兴图新科电子股份有限公司招股意向书，电子信息行业的下游主要包括哪些行业？
7. 武汉兴图新科电子股份有限公司参与的哪个工程荣获了国家科技进步一等奖？
8. 武汉兴图新科电子股份有限公司注册资本是多少？
9. 武汉兴图新科电子股份有限公司法定代表人是谁？
10. 武汉兴图新科电子股份有限公司计划使用本次发行募集资金的多少用于补充流动资金？
```

---

## 第三阶段：DevOps Agent（部署上线）

### 部署目标

- **FastAPI 服务**（uvicorn，端口 8000）
- **Milvus**（Docker 容器，端口 19530，如未部署）
- **Redis** 已部署，直接连接

### docker-compose.yml 骨架

```yaml
version: '3.8'
services:
  milvus:
    image: milvusdb/milvus:latest
    ports:
      - "19530:19530"
    volumes:
      - ./data/milvus_data:/var/lib/milvus
    restart: unless-stopped

  rag-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DEEPSEEK_API_KEY=${DEEPSEEK_API_KEY}
      - MILVUS_HOST=milvus
      - MILVUS_PORT=19530
      - REDIS_HOST=host.docker.internal   # Redis 在宿主机
      - REDIS_PORT=6379
    volumes:
      - D:/models:/models          # bge-m3, bge-reranker 模型挂载
      - ./data:/app/data
    depends_on:
      - milvus
    restart: unless-stopped
```

### 端点

| 端点 | 方法 | 说明 |
|------|------|------|
| `POST /query` | POST | 问答接口，支持 `session_id` 参数实现多轮对话 |
| `POST /query/stream` | POST | 流式问答接口（SSE） |
| `GET /health` | GET | 健康检查（含 Redis + Milvus 连接状态） |
| `POST /upload` | POST | 上传 PDF 文档入库 |

### 本地启动

```bash
# 1. 如 Milvus 未部署，启动 Milvus
docker run -d --name rag-milvus -p 19530:19530 -v D:/Desktop/rag-hermes/rag-project/data/milvus_data:/var/lib/milvus milvusdb/milvus:latest

# 2. 启动 API（Redis 已运行，无需启动）
cd teams/team/project
DEEPSEEK_API_KEY=sk-xxx uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

### 产出物

1. `docker-compose.yml` — 统一编排
2. `Dockerfile` — API 镜像
3. `teams/team/artifacts/ready-for-deploy.md`
4. `teams/team/artifacts/release-notes.md`

---

## 第四阶段：Judge Agent（评测验收）

### 评测维度

| 维度 | 指标 | 目标 |
|------|------|------|
| 检索准确率 | Precision@K | 检索结果中相关文档比例 |
| 检索召回率 | Recall@K | 相关文档中召回了多少 |
| 答案忠实度 | Faithfulness | 回答是否基于检索结果 |
| 答案相关性 | Answer Relevancy | 回答是否回答用户问题 |
| 多轮上下文 | Context Continuity | Redis 记忆是否准确传递历史 |
| 端到端延迟 | Latency | ≤ 3 秒 |
| RAG vs 纯 LLM | 对比分析 | RAG 准确率高于裸 LLM |

### 验收标准

**功能验收：**
- [ ] PDF 解析（Fitz）：准确解析 PDF 中的文字和表格数据
- [ ] 混合检索：向量检索 + BM25 加权融合生效
- [ ] 问答准确性：基于 PDF 的 RAG 回答正确率 ≥ 80%
- [ ] RAG vs 纯 LLM 对比：RAG 回答明显优于裸 LLM
- [ ] Redis 记忆：多轮对话上下文正确传递
- [ ] 交互友好性：界面清晰简洁
- [ ] 多语言支持：中英文问答

**性能验收：**
- [ ] 响应时间 ≤ 3 秒
- [ ] Redis 读写延迟 ≤ 10ms
- [ ] 资源消耗合理，支持长时间运行
- [ ] 容错机制：Redis 断开时降级运行

### 评测方法

1. 使用上面 10 道测试题逐条评测
2. 每条运行 3 次取平均分
3. 分别记录 RAG 版和纯 LLM 版的回答
4. 使用 RAGAS 框架或自定义脚本计算 faithfulness / relevancy 分数
5. 输出对比报告

### 产出物

1. `teams/team/artifacts/evaluation/report.md`
2. `teams/team/artifacts/evaluation/rag_vs_llm_comparison.md`
3. `teams/team/artifacts/evaluation/failures.md`

---

## 交付物清单

| 类别 | 交付物 |
|------|--------|
| 系统功能 | 问答界面（Web UI，支持多轮对话） |
| 系统功能 | 问答引擎（混合检索 + Reranker + LLM 生成） |
| 系统功能 | PDF 解析模块（Fitz / pymupdf） |
| 系统功能 | Redis 短期记忆模块 |
| 系统功能 | 知识库管理模块 |
| 系统功能 | 文档批量入库 CLI |
| 文档 | 技术文档（架构、选型、开发流程） |
| 文档 | 用户手册（操作指南） |
| 演示 | 完整演示视频 |
| 演示 | 10 道测试题的 RAG vs 纯 LLM 对比评测报告 |

---

## 备注

- 工单编号（代码注释用）：`人工智能NLP-RAG-基于PDF文档的问答系统`
- 目标 PDF 文件：《招股说明书.pdf》（需确认存放路径）
- 大模型：**deepseek-v4-pro**，API Key 从环境变量 `DEEPSEEK_API_KEY` 读取
- Embedding 模型：`D:\models\bge-m3`
- Reranker 模型：`D:\models\bge-reranker`
- PDF 解析引擎：**Fitz（pymupdf）**
- 检索方式：**向量检索（Milvus）+ BM25 关键词检索混合（加权融合）**
- 短期记忆：**Redis**（已部署 Docker，端口 6379，直接连接）
- 向量数据库：**Milvus**（如未部署需 `docker run` 启动，端口 19530）
- 配置文件 `config.py` 中模型路径使用 Windows 原生路径 + WSL2 映射双兼容写法
