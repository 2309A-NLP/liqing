# 这一层负责管理“短期记忆”，也就是当前会话的上下文历史。
import json
from typing import Dict, List, Sequence

from .config import get_session_ttl, load_redis


# 这个函数负责从 Redis 中读取最近几轮问答。
# 它会把短期历史拿出来，作为当前回答的上下文补充。
def read_recent_history(session_id: str, limit: int) -> List[Dict[str, str]]:
    # 先连接 Redis。
    redis_cli = load_redis()
    # 每一轮通常有 user 和 assistant 两条，所以这里取 limit*2 条。
    key = f"rag:session:{session_id}:history"
    raw = redis_cli.lrange(key, -limit * 2, -1)
    history: List[Dict[str, str]] = []
    # Redis 里存的是 JSON 字符串，这里需要还原成字典。
    for item in raw:
        try:
            history.append(json.loads(item))
        except json.JSONDecodeError:
            # 如果某条数据坏了，就跳过，不影响整段历史读取。
            continue
    return history


# 这个函数负责把本轮对话写回 Redis。
# 这样下一轮提问时，就还能继续沿用这段短期记忆。
def write_history(session_id: str, user_id: str, role_name: str, question: str, answer: str) -> None:
    # 先连接 Redis。
    redis_cli = load_redis()
    # 当前会话对应的历史键。
    key = f"rag:session:{session_id}:history"
    # 一次对话会存两条记录，保证后续读取时能保持问答结构。
    payloads = [
        {"role": "user", "content": question, "user_id": user_id, "role_name": role_name},
        {"role": "assistant", "content": answer, "user_id": user_id, "role_name": role_name},
    ]
    # pipeline 可以把多个写操作合并执行，减少网络开销。
    pipe = redis_cli.pipeline()
    for payload in payloads:
        pipe.rpush(key, json.dumps(payload, ensure_ascii=False))
    # 给历史设置过期时间，防止旧会话长期占用空间。
    pipe.expire(key, get_session_ttl())
    pipe.execute()


# 这个函数负责把历史列表整理成纯文本。
# 它的目标是把结构化的消息转成适合拼 prompt 的连续文本。
def history_to_text(history: Sequence[Dict[str, str]]) -> str:
    # 把每条历史整理成一行。
    lines: List[str] = []
    # 只保留有内容的记录，空消息没必要放进上下文。
    for item in history:
        role = item.get("role", "")
        content = item.get("content", "")
        if content:
            lines.append(f"{role}: {content}")
    # 最后用换行符拼接成一段连续文本。
    return "\n".join(lines)
