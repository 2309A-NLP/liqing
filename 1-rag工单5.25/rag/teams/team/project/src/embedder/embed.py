"""
Embedding 模块 — 调用 bge-m3 生成向量
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
"""

import os
# suppress TF noise before any import
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from typing import List
from sentence_transformers import SentenceTransformer
from src.config import config


class Embedder:
    """文本向量化器，封装 bge-m3 模型"""

    def __init__(self, model_path: str | None = None):
        self.model_path = model_path or config.BGE_M3_PATH
        self._model: SentenceTransformer | None = None

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            import logging
            import os
            logger = logging.getLogger("rag")

            # 优先 GPU，除非环境变量强制指定 CPU
            device = os.environ.get("SENTENCE_TRANSFORMERS_DEVICE", "")
            if not device:
                device = "cuda" if self._cuda_available() else "cpu"
            logger.info(f"加载 Embedding 模型: {self.model_path} (device={device})")

            try:
                import torch
                model_kwargs = {}
                if device == "cuda":
                    model_kwargs["model_kwargs"] = {
                        "torch_dtype": torch.float16,
                        "attn_implementation": "eager",  # 禁用 flash attention，避免 RTX4060 崩溃
                    }
                self._model = SentenceTransformer(
                    self.model_path,
                    trust_remote_code=True,
                    device=device,
                    **model_kwargs,
                )
            except RuntimeError as e:
                if "out of memory" in str(e).lower() and device == "cuda":
                    logger.warning(f"CUDA OOM，降级到 CPU: {e}")
                    self._model = SentenceTransformer(
                        self.model_path,
                        trust_remote_code=True,
                        device="cpu",
                    )
                else:
                    raise

        return self._model

    @staticmethod
    def _cuda_available() -> bool:
        """检查 CUDA 是否可用"""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    def embed(self, text: str) -> List[float]:
        """单条文本向量化"""
        return self.model.encode(text, normalize_embeddings=True).tolist()

    def embed_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
    ) -> List[List[float]]:
        """批量文本向量化（带进度条）"""
        from tqdm import tqdm
        total = len(texts)
        all_embeddings = []

        with tqdm(total=total, desc="向量化", unit="条", ncols=80) as pbar:
            for i in range(0, total, batch_size):
                batch = texts[i:i + batch_size]
                batch_emb = self.model.encode(
                    batch,
                    batch_size=batch_size,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
                all_embeddings.extend(batch_emb.tolist())
                pbar.update(len(batch))

        return all_embeddings

    @property
    def dimension(self) -> int:
        """返回向量维度"""
        return self.model.get_sentence_embedding_dimension()
