"""
Reranker — bge-reranker 精排（GPU 加速）
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
"""

import os
import logging
from typing import List, Dict, Any
from sentence_transformers import CrossEncoder
from src.config import config

logger = logging.getLogger("rag")


class Reranker:
    """Cross-Encoder 重排序器"""

    def __init__(self, model_path: str | None = None):
        self.model_path = model_path or config.RERANKER_PATH
        self._model: CrossEncoder | None = None

    @property
    def device(self) -> str:
        """检测可用设备"""
        env_device = os.environ.get("SENTENCE_TRANSFORMERS_DEVICE", "")
        if env_device:
            return env_device
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    @property
    def model(self) -> CrossEncoder:
        if self._model is None:
            device = self.device
            logger.info(f"加载 Reranker 模型: {self.model_path} (device={device})")
            try:
                self._model = CrossEncoder(
                    self.model_path,
                    trust_remote_code=True,
                    device=device,
                )
            except RuntimeError as e:
                if "out of memory" in str(e).lower() and device == "cuda":
                    logger.warning(f"Reranker CUDA OOM，降级到 CPU: {e}")
                    self._model = CrossEncoder(
                        self.model_path,
                        trust_remote_code=True,
                        device="cpu",
                    )
                else:
                    raise
        return self._model

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_n: int = 3,
    ) -> List[Dict[str, Any]]:
        """对候选结果精排

        Args:
            query: 原始查询
            candidates: [{"text": str, "page_no": int, ...}, ...]
            top_n: 返回数量

        Returns:
            精排后的 Top-N，含 score
        """
        if not candidates:
            return []

        pairs = [(query, c["text"]) for c in candidates]
        scores = self.model.predict(pairs)

        # 排序
        indexed = list(enumerate(scores))
        indexed.sort(key=lambda x: x[1], reverse=True)
        top = indexed[:top_n]

        results = []
        for idx, score in top:
            cand = dict(candidates[idx])
            cand["score"] = float(score)
            results.append(cand)
        return results
