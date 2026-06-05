# 配置中心负责统一管理环境变量、外部连接和模型加载。
# 这样业务代码就可以专注处理 RAG 流程，而不用到处写配置读取逻辑。
import logging
# 读取系统环境变量时会用到。
import os
# 用于给模型加载函数加缓存，避免重复初始化。
from functools import lru_cache
# 这里主要用于给函数参数和返回值补充类型说明。
from typing import Any, Dict, Optional

# Redis 用于短期历史、长期记忆等缓存型数据存储。
from redis import Redis
# BGE-M3 用于文本向量化，FlagReranker 用于结果重排序。
from FlagEmbedding import BGEM3FlagModel, FlagReranker
# Milvus 连接管理模块，用来连接向量数据库。
from pymilvus import connections

# DeepSeek 采用 OpenAI 兼容接口，所以这里优先导入 OpenAI 客户端。
try:
    # 如果安装了 openai，就使用它的兼容客户端去访问 DeepSeek。
    from openai import OpenAI
except ImportError:  # pragma: no cover
    # 如果没有安装 openai，后续在加载 LLM 客户端时再显式报错。
    OpenAI = None


# 下面这些 DEFAULT_* 是所有配置项的兜底值。
# 如果环境变量没配，就会使用这些默认值继续运行。

# ========== Milvus 向量数据库连接配置 ==========
DEFAULT_MILVUS_HOST = "127.0.0.1"
DEFAULT_MILVUS_PORT = "19530"
DEFAULT_MILVUS_USER = ""
DEFAULT_MILVUS_PASSWORD = ""
DEFAULT_MILVUS_DATABASE = "default"
# ========== Milvus 集合（数据存储容器）配置 ==========
DEFAULT_COLLECTION = "pdf_chunks"
DEFAULT_MEMORY_COLLECTION = "rag_long_term_memory"
# ========== 本地模型路径配置 ==========
DEFAULT_EMBED_MODEL_PATH = r"D:\models\bge-m3"
DEFAULT_RERANK_MODEL_PATH = r"D:\models\bge-reranker-base"
# ========== DeepSeek 大语言模型配置 ==========
DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEFAULT_DEEPSEEK_MODEL = "deepseek-chat"
# ========== Redis 会话存储配置 ==========
DEFAULT_REDIS_URL = "redis://127.0.0.1:6379/0"
DEFAULT_SESSION_TTL = 60 * 60 * 24  # 会话过期时间为 24 小时
# ========== 检索与生成参数配置 ==========
DEFAULT_TOP_K = 8  # 向量检索返回 8 个结果
DEFAULT_BM25_TOP_K = 30  # BM25 关键词检索返回 30 个
DEFAULT_RERANK_TOP_K = 5  # 重排序保留 5 个
DEFAULT_MAX_TOKENS = 1024  # 限制大模型生成 1024 个 token
# ========== 长期对话记忆配置 ==========
DEFAULT_MEMORY_TOP_K = 4
DEFAULT_MEMORY_MAX_ITEMS = 5000
DEFAULT_MEMORY_IMPORTANCE_THRESHOLD = 0.45

# 这个 logger 由当前模块专用，便于定位配置和连接问题。
logger = logging.getLogger(__name__)


# 把多个字符串片段拼成 Redis key，中间用冒号分隔。
def redis_key(*parts: str) -> str:
    return ":".join(parts)


# 读取整数类型的环境变量，并支持上下限保护。
def env_int(name: str, default: int, minimum: Optional[int] = None, maximum: Optional[int] = None) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError:
        value = default
    if minimum is not None:
        value = max(minimum, value)
    if maximum is not None:
        value = min(maximum, value)
    return value


# 加载文本向量模型。
# lru_cache(maxsize=1) 表示同一个进程里只初始化一次。
@lru_cache(maxsize=1)
def load_embedder() -> BGEM3FlagModel:
    return BGEM3FlagModel(os.getenv("EMBED_MODEL_PATH", DEFAULT_EMBED_MODEL_PATH), use_fp16=True)


# 加载重排序模型。
# 重排序模型也很重，所以同样只创建一次。
@lru_cache(maxsize=1)
def load_reranker() -> FlagReranker:
    return FlagReranker(os.getenv("RERANK_MODEL_PATH", DEFAULT_RERANK_MODEL_PATH), use_fp16=True)


# 加载 OpenAI 兼容的 LLM 客户端（这里用于 DeepSeek）。
# 这个函数是对外部大模型服务的统一入口。
@lru_cache(maxsize=1)
def load_llm_client():
    if OpenAI is None:
        raise ImportError("缺少 openai 依赖，请先安装：pip install openai")
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("请先设置环境变量 DEEPSEEK_API_KEY")
    base_url = os.getenv("DEEPSEEK_BASE_URL", DEFAULT_DEEPSEEK_BASE_URL)
    logger.info("initializing llm client base_url=%s model=%s", base_url, get_deepseek_model_name())
    return OpenAI(api_key=api_key, base_url=base_url)


# Redis 客户端也做缓存，避免每次调用都重新建连接。
@lru_cache(maxsize=1)
def load_redis() -> Redis:
    redis_url = os.getenv("REDIS_URL", DEFAULT_REDIS_URL)
    redis_password = os.getenv("REDIS_PASSWORD", "123456")
    kwargs: Dict[str, Any] = {"decode_responses": True}
    if redis_password:
        kwargs["password"] = redis_password
    return Redis.from_url(redis_url, **kwargs)


# 连接 Milvus 向量数据库，供检索和长期记忆使用。
def connect_milvus() -> None:
    host = os.getenv("MILVUS_HOST", DEFAULT_MILVUS_HOST)
    port = os.getenv("MILVUS_PORT", DEFAULT_MILVUS_PORT)
    user = os.getenv("MILVUS_USER", DEFAULT_MILVUS_USER) or None
    password = os.getenv("MILVUS_PASSWORD", DEFAULT_MILVUS_PASSWORD) or None
    db_name = os.getenv("MILVUS_DATABASE", DEFAULT_MILVUS_DATABASE)
    logger.info("connecting milvus host=%s port=%s db=%s", host, port, db_name)
    connections.connect(alias="default", host=host, port=port, user=user, password=password, db_name=db_name)


# 下方这些 get_* 函数统一读取单项配置，方便上层代码调用。
def get_milvus_collection_name() -> str:
    return os.getenv("MILVUS_COLLECTION", DEFAULT_COLLECTION)


def get_memory_collection_name() -> str:
    return os.getenv("MILVUS_MEMORY_COLLECTION", DEFAULT_MEMORY_COLLECTION)


def get_deepseek_model_name() -> str:
    return os.getenv("DEEPSEEK_MODEL", DEFAULT_DEEPSEEK_MODEL)


def get_session_ttl() -> int:
    return env_int("SESSION_TTL", DEFAULT_SESSION_TTL, minimum=60)


def get_memory_max_items() -> int:
    return env_int("MEMORY_MAX_ITEMS", DEFAULT_MEMORY_MAX_ITEMS, minimum=1)


def get_memory_importance_threshold() -> float:
    # 这个阈值决定“什么内容值得作为长期记忆保存”。
    try:
        return float(os.getenv("MEMORY_IMPORTANCE_THRESHOLD", str(DEFAULT_MEMORY_IMPORTANCE_THRESHOLD)))
    except ValueError:
        return DEFAULT_MEMORY_IMPORTANCE_THRESHOLD



