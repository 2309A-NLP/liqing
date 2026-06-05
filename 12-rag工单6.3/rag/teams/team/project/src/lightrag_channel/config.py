"""
LightRAG 通道 — 配置
工单12：LightRAG 优化任务

使用 MIMO-V2.5-Pro 作为 LLM，本地 bge-m3 作为 Embedding。
"""

import os
from pathlib import Path

# ── 项目路径 ──
PROJECT_ROOT = Path(__file__).parent.parent.parent  # teams/team/project/
DATA_DIR = PROJECT_ROOT / "data"
SOURCE_DOCS_DIR = DATA_DIR / "source_docs"
LIGHTRAG_STORAGE_DIR = DATA_DIR / "lightrag_storage"

# ── MIMO LLM 配置 ──
MIMO_BASE_URL = "https://token-plan-cn.xiaomimimo.com/v1"
MIMO_API_KEY = os.environ.get("XIAOMI_API_KEY", "")
MIMO_MODEL = "mimo-v2.5-pro"

# 如果环境变量没设，用 WSL 里 .hermes/.env 的值
if not MIMO_API_KEY:
    _hermes_env = Path(r"\\wsl.localhost\Ubuntu-24.04\home\lqing\.hermes\.env")
    if _hermes_env.exists():
        for line in _hermes_env.read_text(encoding="utf-8").splitlines():
            if line.startswith("XIAOMI_API_KEY="):
                MIMO_API_KEY = line.split("=", 1)[1].strip().strip('"').strip("'")
                break

# 设置 OPENAI_API_KEY 环境变量（LightRAG 内部会读这个）
if MIMO_API_KEY:
    os.environ["OPENAI_API_KEY"] = MIMO_API_KEY
    os.environ["OPENAI_API_BASE"] = MIMO_BASE_URL

# ── Embedding 配置 ──
# 复用现有项目的 bge-m3 模型
_MODEL_DIR = os.environ.get("MODEL_DIR", r"D:\models")
BGE_M3_PATH = os.environ.get(
    "MODEL_BGE_M3_PATH",
    os.path.join(_MODEL_DIR, "bge-m3")
)
EMBEDDING_DIM = 1024  # bge-m3 输出 1024 维

# ── LightRAG 查询参数 ──
DEFAULT_QUERY_MODE = "mix"  # local/global/hybrid/mix/naive
DEFAULT_TOP_K = 10
