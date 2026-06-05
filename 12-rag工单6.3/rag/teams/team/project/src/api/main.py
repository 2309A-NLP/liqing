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
_lightrag_rag = None  # LightRAG 实例
_lightrag_initialized = False


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


async def get_lightrag():
    """获取 LightRAG 实例（异步初始化）"""
    global _lightrag_rag, _lightrag_initialized
    if not _lightrag_initialized:
        from src.lightrag_channel.init_lightrag import create_lightrag_instance
        _lightrag_rag = await create_lightrag_instance()
        _lightrag_initialized = True
    return _lightrag_rag


# ── 智能引擎分类器 ──

def classify_query(question: str) -> str:
    """根据问题内容自动选择最优引擎

    Returns:
        "traditional" — 精确数值、原文引用、图表数据、反向查询
        "lightrag"    — 综合分析、关系推理、跨文档对比
    """
    q = question.strip()

    # 1. 反向/否定查询 → 传统RAG（图谱不支持反向关系）
    negative_patterns = ["不存在", "不是", "不属于", "没有", "除了", "排除", "非"]
    if any(p in q for p in negative_patterns):
        logger.info(f"[分类器] 反向查询 → traditional")
        return "traditional"

    # 2. 精确数值类关键词
    numeric_keywords = [
        "多少", "几", "比例", "金额", "收入", "股数", "注册资本",
        "增长率", "排名第", "第几", "几个", "几家", "几项",
        "发行", "募集", "净利润", "总资产", "净资产", "毛利率",
        "占发行后", "万股", "亿元", "万元",
    ]
    # 3. 综合分析类关键词
    analysis_keywords = [
        "区别", "对比", "比较", "风险", "业务", "主要", "核心",
        "有哪些", "什么产品", "什么业务", "如何", "怎么样",
        "优势", "劣势", "特点", "概述", "介绍", "分析",
    ]

    numeric_score = sum(1 for k in numeric_keywords if k in q)
    analysis_score = sum(1 for k in analysis_keywords if k in q)

    # 4. 提到具体公司名 → 倾向LightRAG（图谱检索更精准）
    has_company = "力源" in q or "兴图" in q
    has_numeric = any(k in q for k in ["多少", "几", "股数", "收入", "金额", "比例"])
    if has_company and analysis_score > 0:
        logger.info(f"[分类器] 公司+分析查询 → lightrag")
        return "lightrag"
    if has_company and has_numeric and analysis_score == 0:
        logger.info(f"[分类器] 公司+数值查询 → traditional")
        return "traditional"

    if analysis_score > numeric_score:
        logger.info(f"[分类器] 综合分析查询 → lightrag (分析={analysis_score}, 数值={numeric_score})")
        return "lightrag"
    elif numeric_score > analysis_score:
        logger.info(f"[分类器] 数值查询 → traditional (数值={numeric_score}, 分析={analysis_score})")
        return "traditional"
    else:
        # 分数相等，默认用传统RAG（更稳）
        logger.info(f"[分类器] 未明确类型 → traditional (默认)")
        return "traditional"


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
    engine = req.engine.lower()

    # 智能引擎选择
    if engine == "auto":
        engine = classify_query(question)
        logger.info(f"[自动选引擎] → {engine}")

    _t0 = time.time()

    if not question:
        raise HTTPException(status_code=400, detail="问题不能为空")

    # 1. 获取历史（计时）
    t_mem = time.time()
    history = get_memory().get_history(session_id)
    mem_ms = (time.time() - t_mem) * 1000
    logger.info(f"[记忆读取] {mem_ms:.0f}ms | 历史轮数: {len(history)}")

    # 2. 根据引擎选择检索方式
    if engine == "lightrag":
        # LightRAG 检索 + 生成
        try:
            from src.lightrag_channel.query import query_lightrag
            from src.lightrag_channel.doc_filter import detect_company, rewrite_question_for_company

            # 文档隔离：检测公司并改写问题
            company = detect_company(question)
            lightrag_question = rewrite_question_for_company(question, company)

            rag = await get_lightrag()
            lightrag_result = await query_lightrag(
                rag, lightrag_question, mode=req.lightrag_mode, top_k=10
            )
            result = {
                "answer": lightrag_result["answer"],
                "sources": [],  # LightRAG 不返回传统 sources
            }
        except Exception as e:
            logger.error(f"LightRAG 查询失败: {e}")
            raise HTTPException(status_code=500, detail=f"LightRAG 查询失败: {str(e)}")
    else:
        # 传统 RAG 流程
        # Query 理解（提取文档过滤）
        source_filter = get_understander().extract_source_filter(question, history)
        if source_filter:
            logger.info(f"[文档过滤] source_filter={source_filter}")

        # 提取检索专用query
        retrieval_query = get_understander().extract_retrieval_query(question)
        if retrieval_query != question:
            logger.info(f"[检索query] {retrieval_query[:60]}")

        # 混合检索
        context_chunks = []
        try:
            context_chunks = get_retriever().retrieve(
                question, source_filter=source_filter, retrieval_query=retrieval_query,
            )
        except Exception as e:
            logger.error(f"检索失败: {e}")
            context_chunks = []

        # 生成回答（计时）
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

    # 3. 写入历史
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

    # 4. 记录查询日志
    elapsed = time.time() - _t0
    sources = result.get("sources", [])
    log_query(
        question=question,
        answer=result["answer"][:200],
        latency=elapsed,
        sources_count=len(sources),
        status="ok",
        session_id=session_id,
        engine=engine,
    )

    # 5. 全链路总耗时
    total_ms = (time.time() - _t0) * 1000
    logger.info(f"[全链路] 总耗时={total_ms:.0f}ms | 引擎={engine} | 记忆={mem_ms:.0f}ms")

    # 6. 返回
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
    engine = req.engine.lower()

    # 智能引擎选择
    if engine == "auto":
        engine = classify_query(question)
        logger.info(f"[自动选引擎] → {engine}")

    _t0 = time.time()

    if not question:
        raise HTTPException(status_code=400, detail="问题不能为空")

    # 1. 记忆读取（计时）
    t_mem = time.time()
    history = get_memory().get_history(session_id)
    mem_ms = (time.time() - t_mem) * 1000
    logger.info(f"[SSE-记忆读取] {mem_ms:.0f}ms | 历史轮数: {len(history)}")

    # 2. 根据引擎选择检索方式
    if engine == "lightrag":
        # LightRAG 不支持流式，一次性返回
        try:
            from src.lightrag_channel.query import query_lightrag
            from src.lightrag_channel.doc_filter import detect_company, rewrite_question_for_company

            # 文档隔离：检测公司并改写问题
            company = detect_company(question)
            lightrag_question = rewrite_question_for_company(question, company)

            rag = await get_lightrag()
            lightrag_result = await query_lightrag(
                rag, lightrag_question, mode=req.lightrag_mode, top_k=10
            )
            answer = lightrag_result["answer"]
        except Exception as e:
            logger.error(f"LightRAG 查询失败: {e}")
            answer = f"LightRAG 查询失败: {str(e)}"

        async def lightrag_stream():
            yield f"data: {json.dumps({'token': answer})}\n\n"
            # 发送引擎信息
            yield f"event: engine\ndata: {json.dumps({'engine': 'lightrag', 'mode': req.lightrag_mode})}\n\n"
            # 记录历史
            get_memory().add_record(session_id=session_id, role="user", content=question)
            get_memory().add_record(session_id=session_id, role="assistant", content=answer)
            yield "event: done\ndata: {}\n\n"

        return StreamingResponse(lightrag_stream(), media_type="text/event-stream")

    # 传统 RAG 流程
    # Query 理解（提取文档过滤）
    source_filter = get_understander().extract_source_filter(question, history)
    if source_filter:
        logger.info(f"[SSE-文档过滤] source_filter={source_filter}")

    # 3. 提取检索专用query（去掉公司名前缀，语义更精准）
    retrieval_query = get_understander().extract_retrieval_query(question)
    if retrieval_query != question:
        logger.info(f"[SSE-检索query] {retrieval_query[:60]}")

    # 4. 混合检索
    context_chunks = []
    try:
        context_chunks = get_retriever().retrieve(
            question, source_filter=source_filter, retrieval_query=retrieval_query,
        )
    except Exception as e:
        logger.error(f"检索失败: {e}")

    # 5. 生成
    generator = get_generator()
    _gen_start = time.time()

    def event_stream():
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
                # 发送引擎信息
                yield f"event: engine\ndata: {json.dumps({'engine': 'traditional'})}\n\n"
                # 记录历史
                get_memory().add_record(session_id=session_id, role="user", content=question)
                get_memory().add_record(session_id=session_id, role="assistant", content="[streamed]", sources=context_chunks)
                yield "event: done\ndata: {}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/upload", response_model=UploadResponse)
async def upload(file: UploadFile = File(...)):
    """上传 PDF — 需先用 MinerU 解析后再通过 ingest.py 入库

    MinerU 是离线解析工具，不支持在线实时解析。
    请按以下步骤操作：
      1. magic-pdf -p <file.pdf> -o data/source_docs/ -m auto
      2. 将生成的 content_list.json 放到 data/source_docs/
      3. python ingest.py
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="仅支持 PDF 文件")

    # 保存文件到 uploads 目录
    upload_dir = config.DATA_DIR / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    file_path = upload_dir / file.filename

    content = await file.read()
    with open(file_path, "wb") as f:
        f.write(content)

    return UploadResponse(
        status="info",
        message=(
            f"文件已保存至 {file_path}。"
            f"请先用 MinerU 解析：magic-pdf -p {file_path} -o data/source_docs/ -m auto，"
            f"然后将 content_list.json 放到 data/source_docs/，运行 python ingest.py 入库。"
        ),
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
