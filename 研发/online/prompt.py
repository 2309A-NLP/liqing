# prompt.py 只是一个统一导出入口，方便上层从这里直接拿到提示词相关函数。
from .retrieval import build_context, build_memory_query, build_prompt, build_prompt_preview

# 这里显式声明对外暴露的函数，避免上层到处去找实现位置。
__all__ = ["build_context", "build_memory_query", "build_prompt", "build_prompt_preview"]
