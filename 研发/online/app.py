from pathlib import Path
import sys

# 先找到项目根目录，确保无论从哪里启动脚本，都能正确导入本地模块。
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  # rag-multi-role/
DEV_DIR = PROJECT_ROOT / "研发"  # 研发/
if str(DEV_DIR) not in sys.path:
    sys.path.insert(0, str(DEV_DIR))

# 这里导入的是整个在线服务的入口应用对象。
from online.api import app


# 只有直接执行这个文件时，才会启动开发服务器。
if __name__ == "__main__":
    import uvicorn

    # 使用 uvicorn 把 FastAPI 应用跑在本机 8002 端口。
    uvicorn.run(app, host="127.0.0.1", port=8002)
