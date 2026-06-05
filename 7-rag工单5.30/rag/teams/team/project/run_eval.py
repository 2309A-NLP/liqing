"""一键跑RAGAS评测：先启动API，等就绪，再跑评测"""
import subprocess
import time
import sys
import os

# 1. 启动API
print("启动RAG API...")
api_proc = subprocess.Popen(
    [sys.executable, "run.py"],
    cwd=os.path.dirname(os.path.abspath(__file__)),
)

# 2. 等API就绪
import httpx
print("等待API就绪...")
for i in range(60):
    time.sleep(2)
    try:
        r = httpx.get("http://127.0.0.1:8004/health", timeout=3)
        print(f"  健康检查: status={r.status_code} (第{i+1}次)")
        if int(r.status_code) == 200:
            print("API就绪!")
            break
    except Exception as e:
        print(f"  等待中... ({i*2}s) {e}")
else:
    print("API启动超时!")
    api_proc.terminate()
    sys.exit(1)

# 3. 跑RAGAS评测
print("\n开始RAGAS评测...")
eval_proc = subprocess.run(
    [sys.executable, "tests/eval_ragas.py", "--port", "8004"],
    cwd=os.path.dirname(os.path.abspath(__file__)),
)

# 4. 关闭API
print("\n关闭API...")
api_proc.terminate()
api_proc.wait(timeout=10)
print("完成!")
