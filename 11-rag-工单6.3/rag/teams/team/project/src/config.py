"""
RAG 问答系统 — 全局配置
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统

所有配置优先从环境变量读取，支持 Windows 原生路径 + WSL2 映射双兼容。
"""

import os
from pathlib import Path


class Config:
    # ── LLM ──
    DEEPSEEK_API_KEY: str = os.environ.get("DEEPSEEK_API_KEY", "")
    DEEPSEEK_BASE_URL: str = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    DEEPSEEK_MODEL: str = os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-flash")

    # ── Milvus ──
    MILVUS_HOST: str = os.environ.get("MILVUS_HOST", "localhost")
    MILVUS_PORT: int = int(os.environ.get("MILVUS_PORT", "19530"))
    MILVUS_COLLECTION: str = "ccf_chunks"

    # ── Redis ──
    REDIS_HOST: str = os.environ.get("REDIS_HOST", "localhost")
    REDIS_PORT: int = int(os.environ.get("REDIS_PORT", "6379"))
    REDIS_TTL: int = 86400  # 24h
    REDIS_HISTORY_N: int = 5  # 最近几轮

    # ── 模型路径（双兼容） ──
    _MODEL_DIR = os.environ.get("MODEL_DIR", r"D:\models")

    @property
    def BGE_M3_PATH(self) -> str:
        path = os.environ.get("MODEL_BGE_M3_PATH", os.path.join(self._MODEL_DIR, "bge-m3"))
        return self._resolve_path(path)

    @property
    def RERANKER_PATH(self) -> str:
        path = os.environ.get("MODEL_RERANKER_PATH", os.path.join(self._MODEL_DIR, "bge-reranker-base"))
        return self._resolve_path(path)

    # ── 分块参数 ──
    CHUNK_SIZE: int = int(os.environ.get("CHUNK_SIZE", "512"))
    CHUNK_OVERLAP: int = int(os.environ.get("CHUNK_OVERLAP", "128"))

    # ── 检索参数 ──
    VECTOR_TOP_K: int = 20
    BM25_TOP_K: int = 20
    HYBRID_TOP_K: int = 10
    RERANK_TOP_N: int = 5
    VECTOR_WEIGHT: float = 0.7
    BM25_WEIGHT: float = 0.3

    # ── Embedding ──
    EMBEDDING_DIM: int = 1024  # bge-m3 输出 1024 维
    EMBEDDING_BATCH_SIZE: int = 32

    # ── Embedding 变体（用于对比评测） ──
    # base:    原始 bge-base-zh-v1.5 (768维)
    # base_ft: 微调后的 bge-base-zh-v1.5 (768维)
    # m3:      bge-m3 (1024维，默认)
    EMBED_VARIANT: str = os.environ.get("EMBED_VARIANT", "m3")

    # 变体 → (模型路径属性名, 集合名, 维度)
    _VARIANT_REGISTRY: dict = {
        "base":    {"path_key": "BGE_BASE_PATH",    "collection": "ccf_chunks_base",    "dim": 768},
        "base_ft": {"path_key": "BGE_BASE_FT_PATH", "collection": "ccf_chunks_base_ft", "dim": 768},
        "m3":      {"path_key": "BGE_M3_PATH",      "collection": "ccf_chunks",         "dim": 1024},
    }

    @property
    def BGE_BASE_PATH(self) -> str:
        """原始 bge-base-zh-v1.5 模型路径"""
        path = os.environ.get("MODEL_BGE_BASE_PATH", os.path.join(self._MODEL_DIR, "bge-base-zh-v1.5"))
        return self._resolve_path(path)

    @property
    def BGE_BASE_FT_PATH(self) -> str:
        """微调后的 bge-base-zh-v1.5 模型路径"""
        path = os.environ.get(
            "MODEL_BGE_BASE_FT_PATH",
            str(Path(__file__).resolve().parent.parent.parent.parent.parent.parent / "eleven_project" / "output" / "bge-base-zh-v1.5-finetuned"),
        )
        return self._resolve_path(path)

    def get_variant_config(self, variant: str | None = None) -> dict:
        """获取变体配置：模型路径、集合名、维度

        Args:
            variant: 变体名 (base/base_ft/m3)，None 则用 EMBED_VARIANT

        Returns:
            {"model_path": str, "collection": str, "dim": int}
        """
        v = variant or self.EMBED_VARIANT
        if v not in self._VARIANT_REGISTRY:
            raise ValueError(f"未知变体: {v}，可选: {list(self._VARIANT_REGISTRY.keys())}")
        reg = self._VARIANT_REGISTRY[v]
        model_path = getattr(self, reg["path_key"])
        return {"model_path": model_path, "collection": reg["collection"], "dim": reg["dim"]}

    # ── 项目路径（自动检测 WSL2 映射） ──
    PROJECT_ROOT: Path = Path(__file__).parent.parent  # teams/team/project/
    DATA_DIR: Path = PROJECT_ROOT / "data"
    MILVUS_DATA_DIR: Path = DATA_DIR / "milvus_data"

    # ── MinerU 输出路径 ──
    # MinerU 解析后的 content_list.json 存放目录（离线入库源文件）
    SOURCE_DOCS_DIR: Path = DATA_DIR / "source_docs"

    # ── 默认 PDF 路径（自动入库用）──
    @property
    def DEFAULT_PDF_PATH(self) -> str:
        # 先找工单里指定的招股说明书
        candidates = [
            self.PROJECT_ROOT / "招股说明书1-无水印.pdf",
            self.PROJECT_ROOT.parent.parent / "招股说明书1-无水印.pdf",  # rag-hermes/
            self.PROJECT_ROOT.parent.parent.parent / "招股说明书1-无水印.pdf",  # Desktop/rag-hermes/
        ]
        # 也支持环境变量覆盖
        env_path = os.environ.get("DEFAULT_PDF_PATH", "")
        if env_path:
            candidates.insert(0, Path(env_path).resolve())
        for p in candidates:
            if p.exists():
                return str(p)
        return ""

    @staticmethod
    def _resolve_path(path: str) -> str:
        """Windows 路径转 WSL2 路径"""
        if os.name == "nt":
            return path
        # 如果在 WSL2 中，D:\ → /mnt/d/
        if ":" in path:
            drive = path[0].lower()
            rest = path[2:].replace("\\", "/")
            return f"/mnt/{drive}{rest}"
        return path


config = Config()
