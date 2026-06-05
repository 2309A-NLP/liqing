"""
LightRAG 通道 — 初始化模块
工单12：LightRAG 优化任务

创建 LightRAG 实例，配置 MIMO LLM + 本地 bge-m3 Embedding。
"""

import os
import sys
from pathlib import Path
from functools import partial

# 禁用 TensorFlow 噪音
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import openai_complete_if_cache
from lightrag.utils import EmbeddingFunc

from .config import (
    MIMO_BASE_URL, MIMO_API_KEY, MIMO_MODEL,
    BGE_M3_PATH, EMBEDDING_DIM, LIGHTRAG_STORAGE_DIR,
)
from .prompts import FINANCIAL_ENTITY_TYPES_GUIDANCE


async def llm_model_func(
    prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
) -> str:
    """MIMO LLM 调用函数（直接用 openai 库，不依赖环境变量）"""
    from openai import AsyncOpenAI
    
    client = AsyncOpenAI(
        api_key=MIMO_API_KEY,
        base_url=MIMO_BASE_URL,
    )
    
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})
    
    # 过滤掉 LightRAG 内部参数，只保留 OpenAI API 认识的参数
    openai_kwargs = {}
    for k in ("temperature", "max_tokens", "top_p", "frequency_penalty", "presence_penalty", "stop", "n", "seed"):
        if k in kwargs:
            openai_kwargs[k] = kwargs[k]
    
    response = await client.chat.completions.create(
        model=MIMO_MODEL,
        messages=messages,
        **openai_kwargs,
    )
    
    return response.choices[0].message.content


async def _bge_m3_embed(texts: list[str]) -> "np.ndarray":
    """本地 bge-m3 Embedding 函数（异步版本）
    
    注意：LightRAG 内部调用 result.size，所以必须返回 numpy array，不能转 list。
    """
    import numpy as np
    from sentence_transformers import SentenceTransformer
    
    # 懒加载模型
    if not hasattr(_bge_m3_embed, "_model"):
        device = "cuda" if _check_cuda() else "cpu"
        _bge_m3_embed._model = SentenceTransformer(
            BGE_M3_PATH, trust_remote_code=True, device=device
        )
    
    model = _bge_m3_embed._model
    embeddings = model.encode(texts, normalize_embeddings=True)
    
    # 必须返回 numpy array（LightRAG 内部用 .size 属性）
    if isinstance(embeddings, np.ndarray):
        return embeddings
    return np.array(embeddings)


def _check_cuda() -> bool:
    """检查 CUDA 是否可用"""
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


async def create_lightrag_instance() -> LightRAG:
    """创建并初始化 LightRAG 实例
    
    Returns:
        已初始化的 LightRAG 实例
    """
    # 确保存储目录存在
    storage_dir = str(LIGHTRAG_STORAGE_DIR)
    os.makedirs(storage_dir, exist_ok=True)
    
    rag = LightRAG(
        working_dir=storage_dir,
        llm_model_func=llm_model_func,
        embedding_func=EmbeddingFunc(
            embedding_dim=EMBEDDING_DIM,
            max_token_size=8192,
            func=_bge_m3_embed,
        ),
        # 金融文档实体类型定制
        addon_params={
            "entity_types_guidance": FINANCIAL_ENTITY_TYPES_GUIDANCE,
        },
    )
    
    await rag.initialize_storages()
    return rag
