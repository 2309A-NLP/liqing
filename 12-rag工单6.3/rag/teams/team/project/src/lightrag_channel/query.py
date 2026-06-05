"""
LightRAG 通道 — 查询模块
工单12：LightRAG 优化任务

封装 LightRAG aquery，统一返回格式，与传统 RAG 通道对齐。
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional

from lightrag import QueryParam

from .config import DEFAULT_QUERY_MODE, DEFAULT_TOP_K

logger = logging.getLogger("lightrag_channel")


async def query_lightrag(
    rag,
    question: str,
    mode: str = DEFAULT_QUERY_MODE,
    top_k: int = DEFAULT_TOP_K,
    only_need_context: bool = False,
) -> Dict[str, Any]:
    """LightRAG 查询
    
    Args:
        rag: LightRAG 实例
        question: 查询问题
        mode: 查询模式 (local/global/hybrid/mix/naive)
        top_k: 返回数量
        only_need_context: 是否只返回上下文（不生成回答）
    
    Returns:
        {
            "answer": str,           # 生成的回答
            "contexts": List[str],   # 检索到的上下文
            "mode": str,             # 使用的查询模式
        }
    """
    param = QueryParam(
        mode=mode,
        top_k=top_k,
        only_need_context=only_need_context,
    )
    
    try:
        result = await rag.aquery(question, param=param)
        
        if only_need_context:
            # 只返回上下文
            return {
                "answer": "",
                "contexts": _extract_contexts(result),
                "mode": mode,
            }
        else:
            # 返回完整回答
            return {
                "answer": str(result),
                "contexts": [],
                "mode": mode,
            }
    
    except Exception as e:
        logger.error(f"LightRAG 查询失败: {e}")
        return {
            "answer": f"ERROR: {e}",
            "contexts": [],
            "mode": mode,
        }


def _extract_contexts(result) -> List[str]:
    """从 LightRAG 上下文结果中提取文本列表
    
    LightRAG 的 only_need_context=True 返回的是拼接后的上下文字符串。
    我们按分隔符拆分成列表。
    """
    if isinstance(result, str):
        # 按段落分割
        contexts = [p.strip() for p in result.split("\n\n") if p.strip() and len(p.strip()) > 20]
        return contexts[:10]  # 最多10个上下文
    elif isinstance(result, list):
        return [str(r) for r in result[:10]]
    else:
        return [str(result)]


async def query_lightrag_with_contexts(
    rag,
    question: str,
    mode: str = DEFAULT_QUERY_MODE,
    top_k: int = DEFAULT_TOP_K,
) -> Dict[str, Any]:
    """查询 LightRAG 并同时获取上下文
    
    先用 only_need_context=True 获取上下文，
    再生成回答。这样 RAGAS 评测可以拿到 contexts。
    
    Args:
        rag: LightRAG 实例
        question: 查询问题
        mode: 查询模式
        top_k: 返回数量
    
    Returns:
        {
            "answer": str,
            "contexts": List[str],
            "mode": str,
        }
    """
    # 先获取上下文
    ctx_result = await query_lightrag(
        rag, question, mode=mode, top_k=top_k, only_need_context=True
    )
    contexts = ctx_result["contexts"]
    
    # 再生成回答
    ans_result = await query_lightrag(
        rag, question, mode=mode, top_k=top_k, only_need_context=False
    )
    
    return {
        "answer": ans_result["answer"],
        "contexts": contexts,
        "mode": mode,
    }
