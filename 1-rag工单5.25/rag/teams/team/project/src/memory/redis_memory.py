"""
Redis 短期记忆模块
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
"""
import json
import time
import os
import logging
from typing import List, Dict, Any, Optional
import redis
from src.config import config

logger = logging.getLogger(__name__)


class RedisMemory:
    """Redis 对话历史管理器"""

    def __init__(
        self,
        host: str | None = None,
        port: int | None = None,
        password: str | None = None,
        ttl: int | None = None,
    ):
        self.host = host or config.REDIS_HOST
        self.port = port or config.REDIS_PORT
        self.password = password or os.environ.get("REDIS_PASSWORD", None)
        self.ttl = ttl or config.REDIS_TTL
        self._client: Optional[redis.Redis] = None
        self._available = True

    def _get_client(self) -> Optional[redis.Redis]:
        """获取 Redis 连接（延迟初始化 + 容错 + 密码降级）"""
        if not self._available:
            return None
        if self._client is None:
            try:
                self._client = redis.Redis(
                    host=self.host,
                    port=self.port,
                    password=self.password,
                    decode_responses=True,
                    socket_connect_timeout=2,
                    socket_timeout=2,
                    retry_on_timeout=False,
                )
                self._client.ping()
            except redis.AuthenticationError as e:
                # 密码不匹配 → 新建连接池，不带密码
                logger.warning(f"Redis 密码无效，无密码重试: {e}")
                self._client = redis.Redis(
                    host=self.host,
                    port=self.port,
                    password=None,
                    connection_pool=None,  # 强制新连接池
                    socket_connect_timeout=2,
                    socket_timeout=2,
                    decode_responses=True,
                )
                self._client.ping()
                self.password = None
                logger.info("  Redis: ✅ 无密码连接成功")
            except (redis.ConnectionError, redis.TimeoutError, OSError) as e:
                logger.warning(f"Redis 连接失败，降级为无记忆模式: {e}")
                self._available = False
                self._client = None
        return self._client

    def add_record(
        self,
        session_id: str,
        role: str,
        content: str,
        sources: List[Dict[str, Any]] | None = None,
    ) -> None:
        """写入一条对话记录"""
        client = self._get_client()
        if client is None:
            return

        record = {
            "role": role,
            "content": content,
            "sources": sources or [],
            "timestamp": time.time(),
        }
        key = f"session:{session_id}:history:{int(time.time() * 1000)}"
        try:
            client.setex(key, self.ttl, json.dumps(record, ensure_ascii=False))
        except redis.RedisError as e:
            logger.warning(f"Redis 写入失败: {e}")

    def get_history(
        self,
        session_id: str,
        recent_n: int | None = None,
    ) -> List[Dict[str, Any]]:
        """获取最近 N 轮对话历史

        Args:
            session_id: 会话 ID
            recent_n: 最近几轮，默认 config.REDIS_HISTORY_N

        Returns:
            [{"role": "user"|"assistant", "content": str, ...}, ...]
        """
        client = self._get_client()
        if client is None:
            return []

        recent_n = recent_n or config.REDIS_HISTORY_N

        try:
            # 扫描该 session 的所有 key
            pattern = f"session:{session_id}:history:*"
            keys = sorted(client.scan_iter(match=pattern), reverse=True)
            keys = keys[:recent_n * 2]  # 多取一些因为会过滤

            history = []
            for key in reversed(keys):
                data = client.get(key)
                if data:
                    history.append(json.loads(data))
            return history
        except redis.RedisError as e:
            logger.warning(f"Redis 读取失败: {e}")
            return []

    def clear_session(self, session_id: str) -> None:
        """清空某会话的所有历史"""
        client = self._get_client()
        if client is None:
            return

        pattern = f"session:{session_id}:history:*"
        try:
            keys = list(client.scan_iter(match=pattern))
            if keys:
                client.delete(*keys)
        except redis.RedisError as e:
            logger.warning(f"Redis 清空失败: {e}")
