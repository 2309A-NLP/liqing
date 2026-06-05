"""
FastAPI 应用入口 — RAG 问答系统
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
"""

import sys
from pathlib import Path

# 修复路径：确保 python src/api/main.py 和 python -m src.api.main 都能跑
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  # src/api/ → project/
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import uuid
import time
import json
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import HTMLResponse, StreamingResponse

from src.config import config
from src.loader.pdf_loader import PDFLoader
from src.chunker.text_splitter import Chunker
from src.api.schemas import QueryRequest, QueryResponse, Source, HealthResponse, UploadResponse
from src.logger import logger, log_query

# ── 模板加载 ──
_TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"

def _load_template(name: str) -> str:
    """加载 HTML 模板文件，失败时返回错误页"""
    path = _TEMPLATES_DIR / name
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return f"<h1>模板未找到: {path}</h1><p>请确认 {name} 存在于 {path}</p>"


# ── 全局组件（延迟初始化 + 按需导入） ──
_memory: "RedisMemory | None" = None
_retriever: "HybridRetriever | None" = None
_generator: "Generator | None" = None
_milvus: "MilvusStore | None" = None
_embedder: "Embedder | None" = None
_understander: "QueryUnderstander | None" = None


def get_memory() -> "RedisMemory":
    global _memory
    if _memory is None:
        from src.memory.redis_memory import RedisMemory
        _memory = RedisMemory()
    return _memory


def get_generator() -> "Generator":
    global _generator
    if _generator is None:
        from src.generator.answer_gen import Generator
        _generator = Generator()
    return _generator


def get_embedder():
    """延迟加载 Embedder（需要 sentence-transformers）"""
    from src.embedder.embed import Embedder
    global _embedder
    if _embedder is None:
        from src.config import config
        logger.info(f"加载 Embedding 模型: {config.BGE_M3_PATH}")
        _embedder = Embedder()
    return _embedder


def get_milvus():
    """延迟加载 MilvusStore（需要 pymilvus）"""
    from src.store.milvus_store import MilvusStore
    global _milvus
    if _milvus is None:
        _milvus = MilvusStore()
        try:
            _milvus.connect()
            logger.info(f"Milvus 已连接，文档数: {_milvus.count()}")
        except Exception as e:
            logger.warning(f"Milvus 连接失败: {e}")
    return _milvus


def get_retriever() -> "HybridRetriever":
    global _retriever
    if _retriever is None:
        from src.retriever.hybrid_retriever import HybridRetriever
        _retriever = HybridRetriever(
            embedder=get_embedder(),
            milvus=get_milvus(),
        )
    return _retriever


def get_understander() -> "QueryUnderstander":
    global _understander
    if _understander is None:
        from src.query.understander import QueryUnderstander
        _understander = QueryUnderstander()
    return _understander


# ── FastAPI ──

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动
    logger.info("RAG 问答系统启动中...")
    if not config.DEEPSEEK_API_KEY:
        logger.warning("DEEPSEEK_API_KEY 未设置，API 调用将失败")
    # 健康检查：确认后端连接
    try:
        mem = get_memory()
        if mem._available:
            logger.info(f"  Redis: ✅ connected ({config.REDIS_HOST}:{config.REDIS_PORT})")
    except Exception:
        logger.warning(f"  Redis: ❌ 不可用（降级为无记忆模式）")
    try:
        m = get_milvus()
        c = m.count()
        logger.info(f"  Milvus: ✅ connected, {c} 条文档")
    except Exception as e:
        logger.warning(f"  Milvus: ❌ 不可用 — {e}")
    logger.info("✅ 启动完成")
    yield
    logger.info("RAG 问答系统关闭")


app = FastAPI(
    title="RAG 问答系统",
    description="基于 PDF 文档的问答系统 — 混合检索 + Reranker + Redis 记忆",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
async def health():
    """健康检查"""
    resp = HealthResponse()

    # Milvus
    try:
        m = get_milvus()
        c = m.count()
        resp.milvus = f"connected, documents: {c}"
    except Exception as e:
        resp.milvus = f"error: {e}"
        resp.status = "degraded"

    # Redis
    mem = get_memory()
    if mem._available:
        resp.redis = "connected"
    else:
        resp.redis = "unavailable (degraded mode)"
        resp.status = "degraded"

    return resp



@app.post("/session/clear")
async def clear_session(session_id: str):
    """清空指定 session 的对话历史"""
    mem = get_memory()
    mem.clear_session(session_id)
    return {"status": "cleared", "session_id": session_id}

@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    """问答接口"""
    session_id = req.session_id
    question = req.question.strip()
    _t0 = time.time()

    if not question:
        raise HTTPException(status_code=400, detail="问题不能为空")

    # 1. 获取历史（计时）
    t_mem = time.time()
    history = get_memory().get_history(session_id)
    mem_ms = (time.time() - t_mem) * 1000
    logger.info(f"[记忆读取] {mem_ms:.0f}ms | 历史轮数: {len(history)}")

    # 2. 混合检索
    context_chunks = []
    try:
        context_chunks = get_retriever().retrieve(question)
    except Exception as e:
        logger.error(f"检索失败: {e}")
        context_chunks = []

    # 3. 生成回答（计时）
    t_gen = time.time()
    try:
        result = get_generator().generate(
            question=question,
            context_chunks=context_chunks,
            history=history,
        )
    except Exception as e:
        logger.error(f"生成失败: {e}")
        log_query(question, status="generate_error", error=str(e), session_id=session_id,
                  latency=time.time() - _t0)
        raise HTTPException(status_code=500, detail=f"生成回答失败: {str(e)}")
    gen_ms = (time.time() - t_gen) * 1000
    logger.info(f"[LLM生成] {gen_ms:.0f}ms | answer={result['answer'][:80]}...")

    # 4. 写入历史
    get_memory().add_record(
        session_id=session_id,
        role="user",
        content=question,
    )
    get_memory().add_record(
        session_id=session_id,
        role="assistant",
        content=result["answer"],
        sources=result.get("sources", []),
    )

    # 5. 记录查询日志
    elapsed = time.time() - _t0
    sources = result.get("sources", [])
    log_query(
        question=question,
        answer=result["answer"][:200],
        latency=elapsed,
        sources_count=len(sources),
        status="ok",
        session_id=session_id,
    )

    # 6. 全链路总耗时
    total_ms = (time.time() - _t0) * 1000
    logger.info(f"[全链路] 总耗时={total_ms:.0f}ms | 记忆={mem_ms:.0f}ms | 生成={gen_ms:.0f}ms")

    # 7. 返回
    return QueryResponse(
        answer=result["answer"],
        sources=[Source(**s) for s in sources],
        session_id=session_id,
    )


@app.post("/query/stream")
async def query_stream(req: QueryRequest):
    """流式问答接口（SSE）"""
    session_id = req.session_id or f"session_{int(time.time())}"
    question = req.question.strip()
    _t0 = time.time()

    if not question:
        raise HTTPException(status_code=400, detail="问题不能为空")

    # 1. 记忆读取（计时）
    t_mem = time.time()
    history = get_memory().get_history(session_id)
    mem_ms = (time.time() - t_mem) * 1000
    logger.info(f"[SSE-记忆读取] {mem_ms:.0f}ms | 历史轮数: {len(history)}")

    # 2. 混合检索
    context_chunks = []
    try:
        context_chunks = get_retriever().retrieve(question)
    except Exception as e:
        logger.error(f"检索失败: {e}")

    generator = get_generator()
    _gen_start = time.time()

    async def event_stream():
        for chunk_type, data in generator.generate_stream(question, context_chunks, history):
            if chunk_type == "source":
                yield f"event: source\ndata: {json.dumps(data)}\n\n"
            elif chunk_type == "token":
                yield f"data: {json.dumps({'token': data})}\n\n"
            elif chunk_type == "done":
                gen_ms = (time.time() - _gen_start) * 1000
                total_ms = (time.time() - _t0) * 1000
                logger.info(f"[SSE-LLM生成] {gen_ms:.0f}ms")
                logger.info(f"[SSE-全链路] 总耗时={total_ms:.0f}ms | 记忆={mem_ms:.0f}ms | 生成={gen_ms:.0f}ms")
                # 记录历史
                get_memory().add_record(session_id=session_id, role="user", content=question)
                get_memory().add_record(session_id=session_id, role="assistant", content="[streamed]", sources=context_chunks)
                yield "event: done\ndata: {}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/upload", response_model=UploadResponse)
async def upload(file: UploadFile = File(...)):
    """上传 PDF 并入库"""
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="仅支持 PDF 文件")

    # 保存文件
    upload_dir = config.DATA_DIR / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    file_path = upload_dir / file.filename

    content = await file.read()
    with open(file_path, "wb") as f:
        f.write(content)

    # 解析 + 分块 + 向量化 + 入库
    try:
        # 解析
        loader = PDFLoader(str(file_path))
        pages = loader.extract_pages()
        logger.info(f"解析完成: {len(pages)} 页")

        # 分块
        chunker = Chunker()
        chunks = chunker.chunk_pages(pages, source_file=file.filename)
        logger.info(f"分块完成: {len(chunks)} 块")

        if not chunks:
            return UploadResponse(
                status="warning",
                message="文档解析成功但未生成有效分块",
                file_name=file.filename,
            )

        # 向量化
        embedder = get_embedder()
        texts = [c["text"] for c in chunks]
        embeddings = embedder.embed_batch(texts)

        # Milvus 入库
        milvus = get_milvus()
        inserted = milvus.insert_chunks(chunks, embeddings)

        # BM25 索引（增量重建）
        bm25 = BM25Index()
        try:
            bm25_index_path = config.DATA_DIR / "bm25_index.pkl"
            if bm25_index_path.exists():
                bm25.load(str(bm25_index_path))
                # 追加新块
                existing_chunks = bm25._chunks  # type: ignore
                existing_chunks.extend(chunks)
                bm25.build_index(existing_chunks)
            else:
                bm25.build_index(chunks)
            bm25.save(str(bm25_index_path))
        except ImportError:
            logger.warning("rank_bm25 未安装，跳过 BM25 索引")

        return UploadResponse(
            status="success",
            message=f"文档入库成功",
            chunks_count=inserted,
            file_name=file.filename,
        )

    except Exception as e:
        logger.error(f"文档入库失败: {e}")
        return UploadResponse(
            status="error",
            message=f"入库失败: {str(e)}",
            file_name=file.filename,
        )


@app.get("/stream", response_class=HTMLResponse)
async def stream_chat():
    """流式聊天界面"""
    return HTMLResponse(_load_template("stream.html"))


@app.get("/", response_class=HTMLResponse)
async def index():
    """聊天界面（默认流式模式）"""
    return HTMLResponse(_load_template("stream.html"))


@app.get("/legacy", response_class=HTMLResponse)
async def legacy():
    """非流式聊天界面"""
    return HTMLResponse(_load_template("index.html"))


def main(port: int = 8000):
    """启动入口"""
    import uvicorn
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=port, reload=False)


if __name__ == "__main__":
    import sys
    port = 8000
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        port = int(sys.argv[1])
