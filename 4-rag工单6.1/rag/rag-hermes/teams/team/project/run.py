"""
RAG 问答系统 — 启动入口
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统

用法：
  python run.py                # 启动在线服务（API）
  python run.py --port 8084    # 指定端口
  python run.py 8001           # 端口被占用时换一个
  python ingest.py              # 离线入库（另开终端跑）
"""

import os
# 上线 TensorFlow / oneDNN 噪音（必须在任何 import 之前）
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# GPU 模式（bge-m3 float16 + 禁用 flash attention 避免崩溃）
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# 上线各种噪音
import warnings
import logging
warnings.filterwarnings("ignore", module="tensorflow")
warnings.filterwarnings("ignore", module="tf_keras")
warnings.filterwarnings("ignore", module="jieba")
warnings.filterwarnings("ignore", message=".*pkg_resources.*")
# TF 日志走 Python logging，直接压到 ERROR 级别
logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("tf_keras").setLevel(logging.ERROR)

import sys
import argparse
from pathlib import Path

# 把项目根目录加入 sys.path
PROJECT_ROOT = Path(__file__).parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.api.main import main

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG 问答系统")
    parser.add_argument("--port", type=int, default=8004, help="服务端口（默认 8004）")
    parser.add_argument("port_arg", nargs="?", type=int, help="端口（位置参数，备用）")
    args = parser.parse_args()

    port = args.port
    if args.port_arg:
        port = args.port_arg

    print(f"🚀 启动 RAG 问答系统 (端口 {port})...")
    print(f"   访问: http://localhost:{port}/")
    main(port=port)
