# 这一层负责“长期记忆”，也就是把值得保留的信息沉淀下来。
# 这样系统下次遇到类似问题时，就能从历史记忆中找回来。
import json
from time import time
from typing import Any, Dict, List, Optional, Sequence

from pymilvus import DataType, MilvusClient, utility

from .config import (
    DEFAULT_MEMORY_COLLECTION,
    get_memory_collection_name,
    get_memory_importance_threshold,
    get_memory_max_items,
    load_redis,
)
from .models import MemoryCandidate
from .retrieval import embed_query, get_collection


# 这个函数负责生成稳定的记忆 ID。
# 你可以把它理解成“同一条记忆的唯一编号”，避免重复写入。
def make_memory_id(session_id: str, user_id: str, memory_text: str) -> str:
    import hashlib

    raw = f"{session_id}:{user_id}:{memory_text.strip()}".encode("utf-8")
    return hashlib.sha1(raw).hexdigest()


# 这个函数负责从 Redis 里读取最近保存的长期记忆。
# 它常用于去重、调试和查看记忆保存效果。
def read_memory_records(limit: int = 100, memory_type: str = "") -> List[Dict[str, Any]]:
    redis_cli = load_redis()
    key = "rag:long_term_memory"
    raw = redis_cli.lrange(key, -limit, -1)
    records: List[Dict[str, Any]] = []
    for item in raw:
        try:
            record = json.loads(item)
        except json.JSONDecodeError:
            continue
        if memory_type and record.get("memory_type") != memory_type:
            continue
        records.append(record)
    return records


# 这个函数负责判断新记忆是否和已有记忆重复。
# 它主要用来防止长期记忆库里出现内容重复的问题。
def is_duplicate_memory(record: Dict[str, Any], recent_records: Sequence[Dict[str, Any]]) -> bool:
    record_id = record.get("id", "")
    record_text = str(record.get("content", "")).strip().lower()
    record_user = str(record.get("user_id", ""))
    for item in recent_records:
        if str(item.get("user_id", "")) != record_user:
            continue
        if record_id and item.get("id") == record_id:
            return True
        if record_text and str(item.get("content", "")).strip().lower() == record_text:
            return True
    return False


# 这个函数负责把记忆先写入 Redis。
# Redis 在这里相当于轻量缓存，便于快速查看最近的长期记忆。
def store_memory_record(record: Dict[str, Any]) -> None:
    redis_cli = load_redis()
    key = "rag:long_term_memory"
    redis_cli.rpush(key, json.dumps(record, ensure_ascii=False))
    current_len = redis_cli.llen(key)
    if current_len > get_memory_max_items():
        redis_cli.ltrim(key, current_len - get_memory_max_items(), -1)


# 这个函数负责创建长期记忆的 Milvus 集合。
# 如果集合已经存在就跳过，否则自动创建表结构和索引。
def ensure_memory_collection(collection_name: str = DEFAULT_MEMORY_COLLECTION, vector_dim: int = 1024) -> None:
    if utility.has_collection(collection_name):
        return

    client = MilvusClient(uri="http://localhost:19530")
    schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field(field_name="id", datatype=DataType.VARCHAR, is_primary=True, max_length=64)
    schema.add_field(field_name="session_id", datatype=DataType.VARCHAR, max_length=64)
    schema.add_field(field_name="user_id", datatype=DataType.VARCHAR, max_length=64)
    schema.add_field(field_name="source", datatype=DataType.VARCHAR, max_length=64)
    schema.add_field(field_name="domain", datatype=DataType.VARCHAR, max_length=32)
    schema.add_field(field_name="title", datatype=DataType.VARCHAR, max_length=128)
    schema.add_field(field_name="role", datatype=DataType.VARCHAR, max_length=64)
    schema.add_field(field_name="keywords", datatype=DataType.VARCHAR, max_length=256)
    schema.add_field(field_name="content", datatype=DataType.VARCHAR, max_length=65535)
    schema.add_field(field_name="vector_text", datatype=DataType.VARCHAR, max_length=65535)
    schema.add_field(field_name="memory_type", datatype=DataType.VARCHAR, max_length=32)
    schema.add_field(field_name="importance", datatype=DataType.FLOAT)
    schema.add_field(field_name="created_at", datatype=DataType.INT64)
    schema.add_field(field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=vector_dim)

    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="embedding",
        index_type="HNSW",
        metric_type="COSINE",
        params={"M": 16, "efConstruction": 200},
    )
    client.create_collection(collection_name=collection_name, schema=schema, index_params=index_params)


# 这个函数负责从当前对话里挑出“值得记住”的内容。
# 它用规则去判断重要性，虽然简单，但非常直观和可控。
def extract_memory_candidate(question: str, answer: str, history_text: str, role_name: str) -> Optional[MemoryCandidate]:
    text = "\n".join([question.strip(), answer.strip(), history_text.strip()]).strip()
    if not text:
        return None

    importance = 0.15
    memory_type = "summary"
    tags: List[str] = []

    preference_markers = ["偏好", "喜欢", "尽量", "请用", "以后", "不要", "更希望", "习惯", "风格"]
    task_markers = ["任务", "计划", "需要", "下一步", "待办", "进度", "安排", "提醒"]
    fact_markers = ["是", "为", "属于", "位于", "包含", "负责", "采用", "名称", "地址"]
    personal_markers = ["我叫", "我的", "我喜欢", "我不喜欢", "我习惯", "请记住", "请保存"]

    if any(marker in text for marker in preference_markers):
        memory_type = "preference"
        importance = 0.8
        tags.append("preference")
    elif any(marker in text for marker in task_markers):
        memory_type = "task"
        importance = 0.72
        tags.append("task")
    elif any(marker in text for marker in fact_markers):
        memory_type = "fact"
        importance = 0.62
        tags.append("fact")
    elif any(marker in text for marker in personal_markers):
        memory_type = "profile"
        importance = 0.78
        tags.append("profile")

    if role_name in {"lawyer", "doctor"}:
        importance += 0.05

    if "记住" in text or "长期" in text or "偏好" in text:
        importance += 0.08

    if len(text) > 180:
        importance += 0.05

    importance = min(1.0, importance)
    if importance < get_memory_importance_threshold():
        return None

    source_text = answer.strip() or question.strip() or history_text.strip()
    memory_text = source_text[:300]
    return MemoryCandidate(memory_type=memory_type, memory_text=memory_text, importance=importance, tags=tags, should_save=True)


# 这个函数负责保存长期记忆。
# 它会先筛选候选项、去重，再写入 Redis 和 Milvus。
def save_long_term_memory(
    session_id: str,
    user_id: str,
    role_name: str,
    question: str,
    answer: str,
    history_text: str,
) -> bool:
    candidate = extract_memory_candidate(question, answer, history_text, role_name)
    if candidate is None or not candidate.should_save:
        return False

    memory_text = candidate.memory_text.strip()
    if not memory_text:
        return False

    memory_id = make_memory_id(session_id, user_id, memory_text)
    collection_name = get_memory_collection_name()

    ensure_memory_collection(collection_name)
    if not utility.has_collection(collection_name):
        return False

    record: Dict[str, Any] = {
        "id": memory_id,
        "session_id": session_id,
        "user_id": user_id,
        "role_name": role_name,
        "memory_type": candidate.memory_type,
        "content": memory_text,
        "vector_text": memory_text,
        "source": "chat",
        "domain": "memory",
        "title": "长期记忆",
        "role": role_name,
        "keywords": ",".join(candidate.tags),
        "importance": float(candidate.importance),
        "created_at": int(time()),
    }
    recent_records = read_memory_records(limit=200, memory_type=candidate.memory_type)
    if is_duplicate_memory(record, recent_records):
        return False

    store_memory_record(record)

    try:
        collection = get_collection(collection_name)
        vector = embed_query(memory_text)
        collection.insert(
            [
                [memory_id],
                [session_id],
                [user_id],
                [record["source"]],
                [record["domain"]],
                [record["title"]],
                [record["role"]],
                [record["keywords"]],
                [record["content"]],
                [record["vector_text"]],
                [record["memory_type"]],
                [record["importance"]],
                [record["created_at"]],
                [vector],
            ]
        )
        collection.flush()
        return True
    except Exception:
        return False
