# 对外统一导出会话记忆和长期记忆相关函数。
# 这样上层代码只需要从一个地方导入即可。
# 该文件本质上是一个便捷的聚合导出层。
from .long_term_memory import read_memory_records, save_long_term_memory
from .session_memory import history_to_text, read_recent_history, write_history

__all__ = [
    "history_to_text",
    "read_recent_history",
    "write_history",
    "read_memory_records",
    "save_long_term_memory",
]
