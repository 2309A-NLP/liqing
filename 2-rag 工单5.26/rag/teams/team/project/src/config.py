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
    DEEPSEEK_MODEL: str = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")

    # ── Milvus ──
    MILVUS_HOST: str = os.environ.get("MILVUS_HOST", "localhost")
    MILVUS_PORT: int = int(os.environ.get("MILVUS_PORT", "19530"))
    MILVUS_COLLECTION: str = "pdf_chunks"

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
    RERANK_TOP_N: int = 3
    VECTOR_WEIGHT: float = 0.7
    BM25_WEIGHT: float = 0.3

    # ── Embedding ──
    EMBEDDING_DIM: int = 1024  # bge-m3 输出 1024 维
    EMBEDDING_BATCH_SIZE: int = 32

    # ── 项目路径（自动检测 WSL2 映射） ──
    PROJECT_ROOT: Path = Path(__file__).parent.parent  # teams/team/project/
    DATA_DIR: Path = PROJECT_ROOT / "data"
    MILVUS_DATA_DIR: Path = DATA_DIR / "milvus_data"

    # ── 默认 PDF 路径（自动入库用） ──
    @property
    def DEFAULT_PDF_PATH(self) -> str:
        # 先找工单里指定的招股说明书
        candidates = [
            self.PROJECT_ROOT / "招股说明书1-无水印.pdf",
            self.PROJECT_ROOT.parent.parent / "招股说明书1-无水印.pdf",  # rag-hermes/
            self.PROJECT_ROOT.parent.parent.parent / "招股说明书1-无水印.pdf",  # Desktop/rag-hermes/
            Path(r"D:\Desktop\专高六工单\RAG 工单\RAG 工单\附件\招股说明书1-无水印.pdf"),
            Path(r"D:\Desktop\专高六工单\RAG 工单\RAG 工单\附件\招股说明书2.pdf"),
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
