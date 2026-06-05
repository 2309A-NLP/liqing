# RAG 问答系统

基于 PDF 文档的问答检索系统，使用混合检索（向量 + BM25）+ Reranker + Redis 短期记忆。

## 架构

```
PDF → Fitz解析 → RecursiveCharacter分块 → bge-m3向量化 → Milvus
                                                              ↓
query → 混合检索(Milvus+BM25) → Reranker → deepseek-v4-pro → answer
                                              ↑
                                         Redis 多轮记忆
```

## 快速开始

### 1. 环境准备

```bash
# 安装依赖
pip install -r requirements.txt

# 确保环境变量（或写入 .env）
export DEEPSEEK_API_KEY=sk-your-key-here
```

### 2. 启动 Redis（如未部署）

```bash
docker run -d --name rag-redis -p 6379:6379 redis:7-alpine
```

### 3. 启动 Milvus（如未部署）

```bash
docker run -d --name rag-milvus -p 19530:19530 \
  -v ./data/milvus_data:/var/lib/milvus \
  milvusdb/milvus:latest
```

### 4. 启动 API

```bash
cd teams/team/project
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

打开浏览器访问 `http://localhost:8000`

### 5. 上传文档

```
POST /upload  — 上传 PDF 文档（自动解析 + 分块 + 向量化 + 入库）
POST /query   — 问答接口（支持多轮对话 session_id）
GET  /health  — 健康检查
```

## 技术栈

| 组件 | 选型                             |
|------|--------------------------------|
| PDF 解析 | Fitz (pymupdf)                 |
| 分块 | RecursiveCharacterTextSplitter |
| Embedding | bge-m3 (1024维)                 |
| 向量库 | Milvus                         |
| 关键词检索 | BM25 (rank_bm25)               |
| Reranker | bge-reranker                   |
| LLM | deepseek-v4-pro                |
| 短期记忆 | Redis                          |
| API | FastAPI + uvicorn              |
