# API 层负责把检索、记忆和大模型这些能力串起来，对外提供可调用接口。
import json
import logging
import random
import re
from concurrent.futures import ThreadPoolExecutor
from time import perf_counter
from uuid import uuid4
from typing import Any, Dict

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from .config import connect_milvus, get_deepseek_model_name, get_memory_collection_name, get_milvus_collection_name
from .llm import call_deepseek, stream_deepseek
from .long_term_memory import ensure_memory_collection, read_memory_records, save_long_term_memory
from .models import ChatRequest, ChatResponse, get_role_config, role_presets_dict
from .retrieval import (
    build_context,
    build_memory_query,
    build_prompt,
    build_prompt_preview,
    collect_retrieval_stats,
    embed_query,
    get_collection,
    merge_candidates,
    rerank,
    search_bm25,
    search_dense,
    search_long_term_memory,
)
from .session_memory import history_to_text, read_recent_history, write_history

# 统一的日志对象，便于记录请求、检索和异常。
logger = logging.getLogger(__name__)
# FastAPI 应用实例，所有在线接口都挂在这里。
app = FastAPI(title="Milvus Hybrid RAG API", version="2.1.0")
# 允许跨域请求，方便浏览器前端直接调用后端。
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
# 默认使用最近 8 轮对话作为短期记忆。
DEFAULT_HISTORY_TURNS = 8
# 长期记忆默认召回条数。
DEFAULT_MEMORY_TOP_K = 4
# 并发执行向量检索、BM25 检索和长期记忆检索。
_RETRIEVAL_EXECUTOR = ThreadPoolExecutor(max_workers=3)


# 启动时先完成基础环境准备。
# 这一步会连接 Milvus，并尝试预加载主知识库和长期记忆集合。
# 你在汇报时可以说：服务启动后先做“自检”和“预热”，这样后续第一次请求不会太慢。
@app.on_event("startup")
def startup_event() -> None:
    connect_milvus()
    try:
        _ = get_collection(get_milvus_collection_name())
    except Exception:
        logger.exception("failed to load main collection")
    try:
        ensure_memory_collection(get_memory_collection_name())
        _ = get_collection(get_memory_collection_name())
    except Exception:
        logger.exception("failed to load memory collection")


# 这是健康检查接口。
# 它不处理业务，只负责告诉外部“服务是否正常启动”。
# 部署、监控、前端初始化时，通常都会先调用它确认服务可用。
@app.get("/health")
def health() -> Dict[str, Any]:
    return {"status": "ok", "service": app.title, "version": app.version}


# 这个函数专门负责把一次请求整理成结构化日志。
# 你可以说它的作用是“给每次问答打标签”，方便后面统计、排错和查请求链路。
def _build_request_log(request_id: str, req: ChatRequest, question: str) -> Dict[str, Any]:
    return {
        "event": "chat_request",
        "request_id": request_id,
        "session_id": req.session_id,
        "user_id": req.user_id,
        "role": req.role_name,
        "question_len": len(question),
    }


# 这个函数负责从用户问题里抽取关键词。
# 你可以把它理解成“先抓重点词，再拿这些词去辅助检索”。
def _extract_keywords(question: str):
    import re

    parts = re.findall(r"[\u4e00-\u9fff]{2,}|[A-Za-z0-9]+", question)
    stopwords = {"什么", "哪些", "如何", "为什么", "可以", "是否", "怎么", "请问", "一下", "主要", "依据", "指标", "内容", "问题", "以及", "请", "帮我", "告诉我", "介绍", "说明"}
    return [p for p in parts if p not in stopwords][:8]


# 这个函数负责把原始问题改写成更适合检索的查询串。
# 它会加入历史上下文、同义词和关键词，让召回更容易命中。
def _rewrite_query(question: str, history_text: str = "") -> str:
    base = question.strip()
    kws = _extract_keywords(base)
    expanded = [base]
    # 如果有历史上下文，也带上一小段，提升多轮对话的召回效果。
    if history_text.strip():
        expanded.append(history_text.strip()[:240])
    query_synonyms = {
        "高血压": ["血压高", "血压升高", "hypertension"],
        "危险分层": ["风险分层", "风险评估", "分层", "风险等级"],
        "单纯收缩期高血压": ["收缩压高", "单纯收缩性高血压", "isolated systolic hypertension"],
        "收缩期": ["收缩压", "收缩期血压"],
        "指标": ["标准", "参数", "依据", "特征"],
    }
    # 同义词扩展可以让同一个问题覆盖更多检索写法。
    for key, syns in query_synonyms.items():
        if key in base:
            expanded.extend(syns)
    expanded.extend(kws)
    seen = set()
    deduped = []
    for item in expanded:
        item = str(item).strip()
        if item and item not in seen:
            seen.add(item)
            deduped.append(item)
    return " | ".join(deduped)


# 这个函数会做一层很轻的相关性判断。
# 它的目标不是百分之百精准，而是先把明显不相关的文档过滤掉。
def _doc_is_relevant(question: str, doc) -> bool:
    text = f"{doc.title} {doc.source} {doc.domain} {doc.keywords} {doc.content} {doc.vector_text}".lower()
    kws = [k.lower() for k in _extract_keywords(question)]
    if not kws:
        return True
    hits = sum(1 for k in kws if k in text)
    return hits >= 1 or any(k in text for k in _extract_keywords(question)[:2])


# 这个函数把文档分成“相关”和“暂时不相关”两组。
# 这样后面既可以只拿相关内容去回答，也可以把无关结果留给调试看。
def _filter_relevant_docs(question: str, docs):
    relevant = []
    irrelevant = []
    for doc in docs or []:
        if _doc_is_relevant(question, doc):
            relevant.append(doc)
        else:
            irrelevant.append(doc)
    return relevant, irrelevant


# 这个函数负责把检索结果整理成一段可读文本。
# 它常用于把检索到的资料直接拼接到 prompt 或最终输出里。
def _format_retrieved_sections(docs) -> str:
    if not docs:
        return "【知识库检索内容】\n无检索结果。"

    lines = ["【知识库检索内容】"]
    for idx, doc in enumerate(docs, start=1):
        content = (doc.content or doc.vector_text or "").strip()
        snippet = content[:300] + ("..." if len(content) > 300 else "")
        title = (doc.title or "未命名").strip()
        source = (doc.source or "未知来源").strip()
        lines.append(f"[{idx}] [{doc.retrieval_source}/{doc.memory_type}] {title} | {source}")
        if snippet:
            lines.append(f"   - {snippet}")
    return "\n".join(lines)


# 这个函数负责把模型回答和检索内容组合起来。
# 它的作用是把“资料依据”和“模型结论”放到同一个输出里，方便用户查看。
def _compose_final_answer(model_answer: str, retrieved_docs, question: str = "") -> str:
    relevant_docs, _ = _filter_relevant_docs(question, retrieved_docs)
    retrieved_text = _format_retrieved_sections(relevant_docs)
    return (
        f"{retrieved_text}\n\n"
        f"【大模型生成内容】\n"
        f"{(model_answer or '').strip()}"
    )


# 这个函数会为前端准备三组检索结果。
# 分成全部、相关和不相关三类之后，页面展示和调试都会更直观。
def _compose_response_contexts(question: str, docs):
    relevant_docs, irrelevant_docs = _filter_relevant_docs(question, docs)
    return {
        "all": [doc.__dict__ for doc in docs],
        "relevant": [doc.__dict__ for doc in relevant_docs],
        "irrelevant": [doc.__dict__ for doc in irrelevant_docs],
    }


# 这个函数会提取当前问题对应的相关文档编号。
# 它常用于前端做高亮、引用标注或调试展示。
def _extract_relevant_doc_numbers(question: str, docs):
    relevant_docs, _ = _filter_relevant_docs(question, docs)
    relevant_ids = {id(doc) for doc in relevant_docs}
    numbers = [idx + 1 for idx, doc in enumerate(docs or []) if id(doc) in relevant_ids]
    return numbers, relevant_docs


# 这个函数给回答补上引用标记。
# 这样用户可以很直观看到这段回答参考了哪些检索资料。
def _add_reference_marks(answer_text: str, relevant_docs) -> str:
    answer_text = (answer_text or '').strip()
    if not relevant_docs:
        return answer_text
    refs = ''.join(f'[{i}]' for i in range(1, len(relevant_docs) + 1))
    if refs and '引用：' not in answer_text:
        answer_text = f'{answer_text}\n\n引用：{refs}'
    return answer_text


# 这个函数把文档对象转成字典列表。
# 这样返回给前端时就可以直接 JSON 序列化，不需要再额外转换。
def _context_list_from_docs(docs):
    return [doc.__dict__ for doc in docs]


# 这个函数专门负责打印结构化日志。
# 你可以说它的作用是把每次请求的关键信息记录下来，方便后面排查和统计。
def _log_request(request_log: Dict[str, Any]) -> None:
    logger.info("rag_request %s", json.dumps(request_log, ensure_ascii=False))


# 这个函数负责在真正检索前，把请求需要的基础信息先准备好。
# 它会读历史、取角色配置、算统计信息，并把这些内容一次性打包返回。
def _prepare_chat(req: ChatRequest):
    start = perf_counter()
    request_id = uuid4().hex
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="question 不能为空")

    # 取主知识库集合，作为本轮检索的基础数据源。
    collection = get_collection(get_milvus_collection_name())
    # 读取当前会话最近的短期历史。
    history = read_recent_history(req.session_id, DEFAULT_HISTORY_TURNS)
    short_history_text = history_to_text(history)
    # 读取角色设定，后面会放进 prompt 里。
    role_config = get_role_config(req.role_name)
    history_messages = len(history)
    history_user_messages = sum(1 for item in history if item.get("role") == "user")
    history_assistant_messages = sum(1 for item in history if item.get("role") == "assistant")
    request_log = _build_request_log(request_id, req, question)
    return start, request_id, question, collection, history, short_history_text, role_config, history_messages, history_user_messages, history_assistant_messages, request_log


# 这个函数是整个在线问答最核心的流程入口。
# 它会完成问题改写、多路检索、合并、重排等完整 RAG 过程。
def _run_pipeline(req: ChatRequest):
    start, request_id, question, collection, history, short_history_text, role_config, history_messages, history_user_messages, history_assistant_messages, request_log = _prepare_chat(req)
    try:
        timings: Dict[str, float] = {}
        query_text = _rewrite_query(build_memory_query(question, req.role_name, short_history_text), short_history_text)

        t0 = perf_counter()
        query_vector = embed_query(query_text)
        timings["embed_ms"] = (perf_counter() - t0) * 1000

        dense_top_k = max(req.top_k, req.rerank_top_k * 2)
        bm25_top_k = max(req.bm25_top_k, req.rerank_top_k * 4)
        memory_top_k = max(2, min(DEFAULT_MEMORY_TOP_K, req.rerank_top_k))

        def _timed_call(name, func, *args):
            start_t = perf_counter()
            result = func(*args)
            return name, result, (perf_counter() - start_t) * 1000

        dense_future = _RETRIEVAL_EXECUTOR.submit(_timed_call, "dense", search_dense, collection, query_vector, dense_top_k)
        bm25_future = _RETRIEVAL_EXECUTOR.submit(_timed_call, "bm25", search_bm25, collection, query_text, bm25_top_k)
        memory_future = _RETRIEVAL_EXECUTOR.submit(_timed_call, "memory", search_long_term_memory, get_memory_collection_name(), query_text, memory_top_k)
        _, dense_docs, timings["dense_ms"] = dense_future.result()
        _, bm25_docs, timings["bm25_ms"] = bm25_future.result()
        _, memory_docs, timings["memory_ms"] = memory_future.result()

        for doc in memory_docs:
            doc.memory_type = "memory"

        t0 = perf_counter()
        candidates = merge_candidates(dense_docs, bm25_docs, memory_docs)
        timings["merge_ms"] = (perf_counter() - t0) * 1000

        t0 = perf_counter()
        reranked = rerank(query_text, candidates, req.rerank_top_k)
        timings["rerank_ms"] = (perf_counter() - t0) * 1000

        retrieval_stats = collect_retrieval_stats(dense_docs, bm25_docs, memory_docs)
        retrieval_stats.reranked = len(reranked)
        prompt_preview = question
        prompt_chars = len(question) + len(short_history_text) + len(query_text)
        timings["total_retrieval_ms"] = sum(timings.values())
        return {
            "start": start,
            "request_id": request_id,
            "question": question,
            "collection": collection,
            "history": history,
            "short_history_text": short_history_text,
            "role_config": role_config,
            "history_messages": history_messages,
            "history_user_messages": history_user_messages,
            "history_assistant_messages": history_assistant_messages,
            "request_log": request_log,
            "timings": timings,
            "prompt_chars": prompt_chars,
            "query_text": query_text,
            "retrieval_breakdown": {
                "dense_count": len(dense_docs),
                "bm25_count": len(bm25_docs),
                "memory_count": len(memory_docs),
                "candidate_count": len(candidates),
                "reranked_count": len(reranked),
            },
            "prompt_preview": prompt_preview,
            "dense_docs": dense_docs,
            "bm25_docs": bm25_docs,
            "memory_docs": memory_docs,
            "candidates": candidates,
            "reranked": reranked,
            "retrieval_stats": retrieval_stats,
        }
    except Exception as exc:
        request_log["status"] = "error"
        request_log["error"] = str(exc)
        request_log["latency_ms"] = int((perf_counter() - start) * 1000)
        _log_request(request_log)
        raise


# 这是非流式聊天接口。
# 你可以说它适合“等答案全部生成完，再一次性返回”的场景。
@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    data = _run_pipeline(req)
    start = data["start"]
    request_id = data["request_id"]
    question = data["question"]
    short_history_text = data["short_history_text"]
    role_config = data["role_config"]
    history = data["history"]
    history_messages = data["history_messages"]
    history_user_messages = data["history_user_messages"]
    history_assistant_messages = data["history_assistant_messages"]
    request_log = data["request_log"]
    timings = data["timings"]
    dense_docs = data["dense_docs"]
    bm25_docs = data["bm25_docs"]
    memory_docs = data["memory_docs"]
    candidates = data["candidates"]
    reranked = data["reranked"]
    retrieval_stats = data["retrieval_stats"]

    try:
        if not reranked:
            answer = "资料不足，无法确定"
            request_log.update(
                {
                    "stage": "no_hit",
                    "retrieved_dense": len(dense_docs),
                    "retrieved_bm25": len(bm25_docs),
                    "retrieved_memory": len(memory_docs),
                    "reranked": 0,
                    "answer_source": "fallback",
                    "answer_len": len(answer),
                }
            )
            request_log.update({f"{k}": round(v, 2) for k, v in timings.items()})
            _log_request(request_log)
            write_history(req.session_id, req.user_id, req.role_name, question, answer)
            try:
                save_long_term_memory(req.session_id, req.user_id, req.role_name, question, answer, short_history_text)
            except Exception:
                logger.exception("memory save failed on no-hit path")
            return ChatResponse(
                session_id=req.session_id,
                user_id=req.user_id,
                role_name=req.role_name,
                answer=answer,
                short_memory=history,
                retrieved=[],
                request_id=request_id,
                latency_ms=int((perf_counter() - start) * 1000),
                retrieval_count=0,
                model_name=get_deepseek_model_name(),
                current_role_config=role_config.model_dump(),
                history_turns=len(history) // 2,
                history_messages=history_messages,
                history_user_messages=history_user_messages,
                history_assistant_messages=history_assistant_messages,
                memory_hit=len(memory_docs) > 0,
                retrieval_stats=retrieval_stats,
                prompt_preview="",
                timings_ms={k: round(v, 2) for k, v in timings.items()},
            )

        composed_contexts = _compose_response_contexts(question, reranked)
        relevant_docs = [doc for doc in reranked if doc.__dict__ in composed_contexts["relevant"]]
        context_docs = relevant_docs if relevant_docs else reranked
        context = build_context(context_docs, max_chars_per_doc=360)
        prompt = build_prompt(question, role_config, short_history_text, context)
        prompt_preview = build_prompt_preview(prompt, 320)
        prompt_chars = len(prompt)
        model_answer, model_name = call_deepseek(prompt, req.max_tokens)
        answer = (model_answer or "").strip()

        write_history(req.session_id, req.user_id, req.role_name, question, answer)
        try:
            saved = save_long_term_memory(req.session_id, req.user_id, req.role_name, question, model_answer, short_history_text)
        except Exception:
            saved = False
            logger.exception("memory save failed")
        request_log.update(
            {
                "stage": "generated",
                "retrieved_dense": len(dense_docs),
                "retrieved_bm25": len(bm25_docs),
                "retrieved_memory": len(memory_docs),
                "candidates": len(candidates),
                "reranked": len(reranked),
                "prompt_chars": prompt_chars,
                "answer_source": model_name,
                "answer_len": len(answer),
                "memory_saved": bool(saved),
                "latency_ms": int((perf_counter() - start) * 1000),
                "status": "ok",
            }
        )
        request_log.update({f"{k}": round(v, 2) for k, v in timings.items()})
        _log_request(request_log)

        return ChatResponse(
            session_id=req.session_id,
            user_id=req.user_id,
            role_name=req.role_name,
            answer=answer,
            short_memory=history,
            retrieved=_context_list_from_docs(reranked),
            request_id=request_id,
            latency_ms=int((perf_counter() - start) * 1000),
            retrieval_count=len(reranked),
            model_name=model_name,
            current_role_config=role_config.model_dump(),
            history_turns=len(history) // 2,
            history_messages=history_messages,
            history_user_messages=history_user_messages,
            history_assistant_messages=history_assistant_messages,
            memory_hit=len(memory_docs) > 0,
            retrieval_stats=retrieval_stats,
            prompt_preview=prompt_preview,
            timings_ms={k: round(v, 2) for k, v in timings.items()},
        )
    except HTTPException:
        request_log["status"] = "http_error"
        _log_request(request_log)
        raise
    except Exception as exc:
        request_log["status"] = "error"
        request_log["error"] = str(exc)
        request_log["latency_ms"] = int((perf_counter() - start) * 1000)
        _log_request(request_log)
        raise HTTPException(status_code=500, detail=f"处理失败: {exc}")


# 这是流式聊天接口。
# 它适合前端边接收边展示，用户会更快看到模型输出。
@app.post("/chat/stream")
def chat_stream(req: ChatRequest):
    data = _run_pipeline(req)
    start = data["start"]
    request_id = data["request_id"]
    question = data["question"]
    short_history_text = data["short_history_text"]
    role_config = data["role_config"]
    history = data["history"]
    history_messages = data["history_messages"]
    history_user_messages = data["history_user_messages"]
    history_assistant_messages = data["history_assistant_messages"]
    request_log = data["request_log"]
    timings = data["timings"]
    dense_docs = data["dense_docs"]
    bm25_docs = data["bm25_docs"]
    memory_docs = data["memory_docs"]
    candidates = data["candidates"]
    reranked = data["reranked"]
    retrieval_stats = data["retrieval_stats"]
    composed_contexts = _compose_response_contexts(question, reranked)
    relevant_docs = [doc for doc in reranked if doc.__dict__ in composed_contexts["relevant"]]
    context_docs = relevant_docs if relevant_docs else reranked
    context = build_context(context_docs, max_chars_per_doc=360)
    prompt = build_prompt(question, role_config, short_history_text, context)
    prompt_preview = build_prompt_preview(prompt, 320)
    prompt_chars = len(prompt)

    def event_stream():
        relevant_docs, irrelevant_docs = _filter_relevant_docs(question, reranked)
        context_docs = relevant_docs if relevant_docs else reranked
        meta = {
            "type": "meta",
            "request_id": request_id,
            "session_id": req.session_id,
            "user_id": req.user_id,
            "role_name": req.role_name,
            "latency_ms": int((perf_counter() - start) * 1000),
            "current_role_config": role_config.model_dump(),
            "history_turns": len(history) // 2,
            "history_messages": history_messages,
            "history_user_messages": history_user_messages,
            "history_assistant_messages": history_assistant_messages,
            "memory_hit": len(memory_docs) > 0,
            "retrieval_stats": retrieval_stats.model_dump(),
            "retrieved_all": _context_list_from_docs(reranked),
            "retrieved_relevant": _context_list_from_docs(relevant_docs),
            "retrieved_irrelevant": _context_list_from_docs(irrelevant_docs),
            "prompt_preview": prompt_preview,
        }
        yield f"data: {json.dumps(meta, ensure_ascii=False)}\n\n"
        if not reranked:
            answer = "资料不足，无法确定"
            yield f"data: {json.dumps({'type': 'meta', 'request_id': request_id, 'session_id': req.session_id, 'user_id': req.user_id, 'role_name': req.role_name, 'latency_ms': int((perf_counter() - start) * 1000), 'current_role_config': role_config.model_dump(), 'history_turns': len(history) // 2, 'history_messages': history_messages, 'history_user_messages': history_user_messages, 'history_assistant_messages': history_assistant_messages, 'memory_hit': len(memory_docs) > 0, 'retrieval_stats': retrieval_stats.model_dump(), 'retrieved_all': _context_list_from_docs(reranked), 'retrieved_relevant': [], 'retrieved_irrelevant': _context_list_from_docs(reranked), 'prompt_preview': prompt_preview}, ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps({'type': 'delta', 'content': answer}, ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps({'type': 'done', 'answer': answer, 'request_id': request_id, 'model_name': get_deepseek_model_name(), 'memory_saved': False, 'prompt_preview': prompt_preview, 'retrieved_all': _context_list_from_docs(reranked), 'retrieved_relevant': [], 'retrieved_irrelevant': _context_list_from_docs(reranked)}, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
            return

        answer_chunks = []
        model_name = get_deepseek_model_name()
        relevant_docs, irrelevant_docs = _filter_relevant_docs(question, reranked)
        try:
            for chunk, model in stream_deepseek(prompt, req.max_tokens):
                model_name = model
                if chunk:
                    answer_chunks.append(chunk)
                    yield f"data: {json.dumps({'type': 'delta', 'content': chunk}, ensure_ascii=False)}\n\n"
            model_answer = "".join(answer_chunks)
            answer = (model_answer or "").strip()
            write_history(req.session_id, req.user_id, req.role_name, question, answer)
            try:
                saved = save_long_term_memory(req.session_id, req.user_id, req.role_name, question, model_answer, short_history_text)
            except Exception:
                saved = False
                logger.exception("memory save failed")
            request_log.update(
                {
                    "stage": "stream_generated",
                    "retrieved_dense": len(dense_docs),
                    "retrieved_bm25": len(bm25_docs),
                    "retrieved_memory": len(memory_docs),
                    "candidates": len(candidates),
                    "reranked": len(reranked),
                    "prompt_chars": len(prompt),
                    "answer_source": model_name,
                    "answer_len": len(answer),
                    "memory_saved": bool(saved),
                    "latency_ms": int((perf_counter() - start) * 1000),
                    "status": "ok",
                }
            )
            request_log.update({f"{k}": round(v, 2) for k, v in timings.items()})
            _log_request(request_log)
            yield f"data: {json.dumps({'type': 'done', 'answer': answer, 'request_id': request_id, 'model_name': model_name, 'memory_saved': bool(saved), 'prompt_preview': prompt_preview, 'retrieved_all': _context_list_from_docs(reranked), 'retrieved_relevant': _context_list_from_docs(relevant_docs), 'retrieved_irrelevant': _context_list_from_docs(irrelevant_docs)}, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as exc:
            request_log["status"] = "error"
            request_log["error"] = str(exc)
            request_log["latency_ms"] = int((perf_counter() - start) * 1000)
            _log_request(request_log)
            yield f"data: {json.dumps({'type': 'done', 'answer': '', 'request_id': request_id, 'model_name': model_name, 'error': str(exc)}, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# 清空某个会话的短期历史，让对话从新上下文开始。
# 这是会话重置接口。
# 它的作用是清空某个 session 的短期历史，重新开始一轮对话。
@app.post("/session/reset")
def reset_session(session_id: str) -> Dict[str, str]:
    from .config import load_redis, redis_key

    redis_cli = load_redis()
    redis_cli.delete(redis_key("rag", "session", session_id, "history"))
    return {"status": "ok", "session_id": session_id}


# 返回系统内置的角色模板，供前端选择使用。
# 这是角色模板接口。
# 前端可以通过它拿到内置角色配置，比如 assistant、lawyer、doctor。
@app.get("/role/presets")
def role_presets() -> Dict[str, Dict[str, Any]]:
    return role_presets_dict()


# 查询长期记忆列表，便于调试和查看已保存内容。
# 这是长期记忆查看接口。
# 它可以把已经保存下来的长期记忆记录列出来，方便调试和查看效果。
@app.get("/memory/list")
def list_memory(limit: int = 50, memory_type: str = "") -> Dict[str, Any]:
    limit = max(1, min(limit, 500))
    records = read_memory_records(limit=limit, memory_type=memory_type)
    return {"count": len(records), "limit": limit, "memory_type": memory_type, "items": records}


# 根据一段知识内容，自动生成一个更像问题的句子。
def _random_question_from_text(text: str) -> str:
    raw = re.sub(r"\s+", "", str(text or "").strip())
    if not raw:
        return "这个知识点怎么理解？"
    raw = re.sub(r"^[【\[][^】\]]*[】\]]", "", raw)
    raw = raw.replace("用户问题：", "").replace("问题：", "").replace("结论：", "")
    raw = raw.replace("title:", "").replace("题目：", "").replace("摘要：", "")
    raw = raw.strip("。.!！？?：:、，,")
    if any(k in raw for k in ["如何", "怎么", "什么", "是否", "哪些", "为什么"]):
        return raw if raw.endswith("？") else f"{raw}？"
    if len(raw) > 12:
        raw = raw[:12]
    templates = [
        f"{raw}一般怎么理解？",
        f"{raw}通常怎么判断？",
        f"{raw}怎么处理？",
        f"{raw}需要注意什么？",
        f"{raw}由谁负责？",
    ]
    return random.choice(templates)


# 随机抽取一些知识点，并自动改成问题，适合做演示或练习。
# 这是随机知识接口。
# 它会从知识库里随机抽一些内容，并转成问题形式，适合演示或练习。
@app.get("/knowledge/random")
def random_knowledge(limit: int = Query(4, ge=1, le=12), category: str = Query("all")) -> Dict[str, Any]:
    collection = get_collection(get_milvus_collection_name())
    query_result = collection.query(
        expr="id != ''",
        output_fields=["id", "source", "domain", "title", "role", "keywords", "content", "vector_text"],
        limit=4096,
    )
    items = []
    for row in query_result or []:
        domain = str(row.get("domain", "")).strip()
        if category != "all" and domain != category:
            continue
        content = str(row.get("content") or row.get("vector_text") or row.get("title") or "").strip()
        if not content:
            continue
        question = _random_question_from_text(content)
        items.append({
            "id": str(row.get("id", "")),
            "question": question,
            "content": content,
            "domain": domain,
        })
    random.shuffle(items)
    return {"category": category, "count": len(items), "items": items[:limit]}


