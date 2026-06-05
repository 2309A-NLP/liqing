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

# ── 变体组件注册表（base/base_ft 对比评测用） ──
_variant_components: dict = {}  # variant → {"embedder", "milvus", "retriever"}


def get_variant_components(variant: str) -> dict:
    """获取指定变体的组件（懒加载）

    仅用于 base/base_ft 对比评测。m3 变体直接用全局组件。
    """
    if variant in _variant_components:
        return _variant_components[variant]

    from src.embedder.embed import Embedder
    from src.store.milvus_store import MilvusStore
    from src.retriever.hybrid_retriever import HybridRetriever

    vc = config.get_variant_config(variant)
    logger.info(f"[变体] 初始化 {variant}: 模型={vc['model_path']} 集合={vc['collection']} dim={vc['dim']}")

    embedder = Embedder(model_path=vc["model_path"])
    milvus = MilvusStore(collection_name=vc["collection"])
    try:
        milvus.connect()
        logger.info(f"[变体] {variant} Milvus 连接成功, 文档数: {milvus.count()}")
    except Exception as e:
        logger.warning(f"[变体] {variant} Milvus 连接失败: {e}")

    retriever = HybridRetriever(embedder=embedder, milvus=milvus)
    comps = {"embedder": embedder, "milvus": milvus, "retriever": retriever}
    _variant_components[variant] = comps
    return comps


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
    variant = req.variant
    _t0 = time.time()

    if not question:
        raise HTTPException(status_code=400, detail="问题不能为空")

    # 根据 variant 选择 retriever
    if variant and variant != "m3":
        _comps = get_variant_components(variant)
        _retriever = _comps["retriever"]
        logger.info(f"[变体查询] variant={variant}")
    else:
        _retriever = get_retriever()

    # 1. 获取历史（计时）
    t_mem = time.time()
    history = get_memory().get_history(session_id)
    mem_ms = (time.time() - t_mem) * 1000
    logger.info(f"[记忆读取] {mem_ms:.0f}ms | 历史轮数: {len(history)}")

    # 2. Query 理解（提取文档过滤）
    source_filter = get_understander().extract_source_filter(question, history)
    if source_filter:
        logger.info(f"[文档过滤] source_filter={source_filter}")

    # 3. 提取检索专用query
    retrieval_query = get_understander().extract_retrieval_query(question)
    if retrieval_query != question:
        logger.info(f"[检索query] {retrieval_query[:60]}")

    # 4. 查询分解（对比类问题拆成多个子查询）
    sub_queries = get_understander().decompose_compare_query(question)
    is_compare = len(sub_queries) > 1

    # 判断是否是复合类问题（需要更多数据）
    is_compound = any(sq.get("sub_intent") in ["list_entities", "each_entity_status"] for sq in sub_queries)

    # 5. 混合检索（支持查询分解）
    context_chunks = []
    try:
        if is_compare or is_compound:
            # 对比类/复合类问题：分别检索每个子查询，合并结果
            logger.info(f"[查询分解] 检测到{'复合类' if is_compound else '对比类'}问题，分解为 {len(sub_queries)} 个子查询")
            all_chunks = []
            seen_texts = set()  # 用于去重

            # 复合类问题需要更多数据
            per_query_top_k = 10 if is_compound else 5
            final_top_k = 15 if is_compound else 10

            for sq in sub_queries:
                sq_query = sq["query"]
                sq_filter = sq.get("source_file")
                sq_company = sq.get("company")
                sq_intent = sq.get("sub_intent", "")

                logger.info(f"[查询分解] 检索子查询: {sq_query} (filter={sq_filter}, intent={sq_intent})")

                # 提取子查询的检索query
                sq_retrieval_query = get_understander().extract_retrieval_query(sq_query)

                # 检索
                sq_chunks = _retriever.retrieve(
                    sq_query,
                    source_filter=sq_filter,
                    retrieval_query=sq_retrieval_query,
                    top_k=per_query_top_k,
                )

                # 去重合并
                for chunk in sq_chunks:
                    text_key = chunk.get("text", "")[:100]
                    if text_key not in seen_texts:
                        seen_texts.add(text_key)
                        all_chunks.append(chunk)

            # 按score排序，取top-N
            all_chunks.sort(key=lambda x: x.get("score", 0), reverse=True)
            context_chunks = all_chunks[:final_top_k]

            logger.info(f"[查询分解] 合并结果: {len(context_chunks)} 个chunks")
        else:
            # 普通问题：直接检索
            context_chunks = _retriever.retrieve(
                question, source_filter=source_filter, retrieval_query=retrieval_query,
            )
    except Exception as e:
        logger.error(f"检索失败: {e}")
        context_chunks = []

    # 6. 生成回答（计时）
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
    logger.info(f"[LLM生成] {gen_ms:.0f}ms | answer={result['answer'][:200]}")
    logger.info(f"[答案全文] {result['answer']}")

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
    variant = req.variant
    _t0 = time.time()

    if not question:
        raise HTTPException(status_code=400, detail="问题不能为空")

    # 根据 variant 选择 retriever
    if variant and variant != "m3":
        _comps = get_variant_components(variant)
        _retriever = _comps["retriever"]
        logger.info(f"[变体查询-stream] variant={variant}")
    else:
        _retriever = get_retriever()

    # 1. 记忆读取（计时）
    t_mem = time.time()
    history = get_memory().get_history(session_id)
    mem_ms = (time.time() - t_mem) * 1000
    logger.info(f"[SSE-记忆读取] {mem_ms:.0f}ms | 历史轮数: {len(history)}")

    # 2. Query 理解（提取文档过滤）
    source_filter = get_understander().extract_source_filter(question, history)
    if source_filter:
        logger.info(f"[SSE-文档过滤] source_filter={source_filter}")

    # 3. 提取检索专用query（去掉公司名前缀，语义更精准）
    retrieval_query = get_understander().extract_retrieval_query(question)
    if retrieval_query != question:
        logger.info(f"[SSE-检索query] {retrieval_query[:60]}")

    # 4. 查询分解（对比类问题拆成多个子查询）
    sub_queries = get_understander().decompose_compare_query(question)
    is_compare = len(sub_queries) > 1

    # 判断是否是复合类问题（需要更多数据）
    is_compound = any(sq.get("sub_intent") in ["list_entities", "each_entity_status"] for sq in sub_queries)

    # 5. 混合检索（支持查询分解）
    context_chunks = []
    try:
        if is_compare or is_compound:
            # 对比类/复合类问题：分别检索每个子查询，合并结果
            logger.info(f"[SSE-查询分解] 检测到{'复合类' if is_compound else '对比类'}问题，分解为 {len(sub_queries)} 个子查询")
            all_chunks = []
            seen_texts = set()  # 用于去重

            # 复合类问题需要更多数据
            per_query_top_k = 10 if is_compound else 5
            final_top_k = 15 if is_compound else 10

            for sq in sub_queries:
                sq_query = sq["query"]
                sq_filter = sq.get("source_file")
                sq_company = sq.get("company")
                sq_intent = sq.get("sub_intent", "")

                logger.info(f"[SSE-查询分解] 检索子查询: {sq_query} (filter={sq_filter}, intent={sq_intent})")

                # 提取子查询的检索query
                sq_retrieval_query = get_understander().extract_retrieval_query(sq_query)

                # 检索
                sq_chunks = _retriever.retrieve(
                    sq_query,
                    source_filter=sq_filter,
                    retrieval_query=sq_retrieval_query,
                    top_k=per_query_top_k,
                )

                # 去重合并
                for chunk in sq_chunks:
                    text_key = chunk.get("text", "")[:100]
                    if text_key not in seen_texts:
                        seen_texts.add(text_key)
                        all_chunks.append(chunk)

            # 按score排序，取top-N
            all_chunks.sort(key=lambda x: x.get("score", 0), reverse=True)
            context_chunks = all_chunks[:final_top_k]

            logger.info(f"[SSE-查询分解] 合并结果: {len(context_chunks)} 个chunks")
        else:
            # 普通问题：直接检索
            context_chunks = _retriever.retrieve(
                question, source_filter=source_filter, retrieval_query=retrieval_query,
            )
    except Exception as e:
        logger.error(f"检索失败: {e}")

    # 6. 生成
    generator = get_generator()
    _gen_start = time.time()

    async def event_stream():
        full_answer = []
        for chunk_type, data in generator.generate_stream(question, context_chunks, history):
            if chunk_type == "source":
                yield f"event: source\ndata: {json.dumps(data)}\n\n"
            elif chunk_type == "token":
                full_answer.append(data)
                yield f"data: {json.dumps({'token': data})}\n\n"
            elif chunk_type == "done":
                gen_ms = (time.time() - _gen_start) * 1000
                total_ms = (time.time() - _t0) * 1000
                answer_text = "".join(full_answer)
                logger.info(f"[SSE-LLM生成] {gen_ms:.0f}ms")
                logger.info(f"[答案全文] {answer_text}")
                logger.info(f"[SSE-全链路] 总耗时={total_ms:.0f}ms | 记忆={mem_ms:.0f}ms | 生成={gen_ms:.0f}ms")
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
