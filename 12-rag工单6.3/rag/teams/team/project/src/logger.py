"""
RAG 问答系统 — 日志模块
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统

用法：
  from src.logger import logger, log_query
  logger.info("xxx")
  log_query(question, latency=1.2, sources=3, status="ok")
"""

import os
import sys
import json
import logging
import logging.handlers
from datetime import datetime
from pathlib import Path
from typing import Optional


_LOG_DIR = None


def ensure_log_dir() -> Path:
    """确保日志目录存在"""
    global _LOG_DIR
    if _LOG_DIR is None:
        # 优先用环境变量
        env_dir = os.environ.get("RAG_LOG_DIR", "")
        if env_dir:
            _LOG_DIR = Path(env_dir)
        else:
            _LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
        _LOG_DIR.mkdir(parents=True, exist_ok=True)
    return _LOG_DIR


def setup_logger(name: str = "rag") -> logging.Logger:
    """配置并返回 logger

    输出目标：
      - stdout: INFO+ （彩色，开发友好）
      - 文件:    DEBUG+ （轮转，保留 7 天）
    """
    log_dir = ensure_log_dir()
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    # ── 格式化 ──
    file_fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-5s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-5s | %(message)s",
        datefmt="%H:%M:%S",
    )

    # ── 文件 handler（轮转，10MB × 保留7天） ──
    file_handler = logging.handlers.RotatingFileHandler(
        filename=log_dir / f"{name}.log",
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=7,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_fmt)
    logger.addHandler(file_handler)

    # ── 控制台 handler（INFO+） ──
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_fmt)
    logger.addHandler(console_handler)

    return logger


def log_query(
    question: str,
    answer: Optional[str] = None,
    latency: Optional[float] = None,
    sources_count: int = 0,
    status: str = "ok",
    session_id: str = "",
    error: Optional[str] = None,
    engine: str = "traditional",
) -> None:
    """记录一条查询日志（JSON 格式，方便分析）

    输出到单独的 queries.log 文件，每行一个 JSON。
    """
    log_dir = ensure_log_dir()
    query_log_path = log_dir / "queries.jsonl"

    record = {
        "timestamp": datetime.now().isoformat(),
        "session_id": session_id,
        "question": question[:500],  # 截断防止撑爆
        "sources_count": sources_count,
        "latency_ms": round(latency * 1000, 1) if latency else None,
        "status": status,
        "error": error[:500] if error else None,
        "engine": engine,
    }

    with open(query_log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # 同时在主日志打一条摘要
    logger = logging.getLogger("rag")
    if error:
        logger.error(f"QUERY [{status}] {question[:80]}... → {error}")
    else:
        latency_str = f" [{latency*1000:.0f}ms]" if latency else ""
        logger.info(f"QUERY [{status}]{latency_str} {question[:80]}... → {sources_count} sources")


# ── 默认 logger ──
logger = setup_logger()
