"""一键跑RAGAS对比评测：先启动API，等就绪，再跑评测"""
import subprocess
import time
import sys
import os
import json
from pathlib import Path

# 1. 启动API
print("启动RAG API...")
api_proc = subprocess.Popen(
    [sys.executable, "run.py"],
    cwd=os.path.dirname(os.path.abspath(__file__)),
)

# 2. 等API就绪
import socket
print("等待API就绪...")
api_ready = False
for i in range(60):
    time.sleep(2)
    try:
        sock = socket.create_connection(("127.0.0.1", 8004), timeout=3)
        sock.close()
        print(f"API就绪! (第{i+1}次检查)")
        api_ready = True
        break
    except Exception:
        print(f"  等待中... ({(i+1)*2}s)")

if not api_ready:
    print("API启动超时!")
    api_proc.terminate()
    sys.exit(1)

# 3. 跑RAGAS对比评测（直接调用，不再起子进程）
print("\n开始RAGAS对比评测...")
try:
    # 把项目目录加入path
    project_dir = os.path.dirname(os.path.abspath(__file__))
    if project_dir not in sys.path:
        sys.path.insert(0, project_dir)

    from tests.eval_compare import (
        load_ground_truth,
        run_single_variant,
        generate_comparison_report,
    )

    # 找 ground_truth.json
    gt_path = Path(project_dir).parent.parent.parent.parent / "artifacts" / "evaluation" / "ground_truth.json"
    if not gt_path.exists():
        gt_path = Path(project_dir).parent.parent / "artifacts" / "evaluation" / "ground_truth.json"
    if not gt_path.exists():
        print(f"❌ ground_truth.json 不存在: {gt_path}")
    else:
        gt_data = load_ground_truth(str(gt_path))
        print(f"加载 {len(gt_data)} 道测试题")

        variants = ["base", "base_ft"]
        results = []
        for variant in variants:
            r = run_single_variant(variant, gt_data, 8004)
            results.append(r)

        output_dir = Path(project_dir).parent.parent.parent.parent / "artifacts" / "evaluation"
        generate_comparison_report(results, output_dir)

        # 打印对比摘要
        print(f"\n{'=' * 60}")
        print("对比摘要")
        print(f"{'=' * 60}")
        if len(results) >= 2:
            metric_names = list(results[0]["scores"].keys())
            for metric in metric_names:
                vals = [r["scores"].get(metric, 0) for r in results]
                diff_str = ""
                if len(vals) == 2 and vals[0] > 0:
                    diff = (vals[1] - vals[0]) / vals[0] * 100
                    sign = "+" if diff > 0 else ""
                    diff_str = f" ({sign}{diff:.1f}%)"
                print(f"  {metric:35s}: {vals[0]:.4f} → {vals[1]:.4f}{diff_str}")
        print(f"{'=' * 60}")
except Exception as e:
    import traceback
    print(f"评测出错: {e}")
    traceback.print_exc()

# 4. 关闭API
print("\n关闭API...")
api_proc.terminate()
api_proc.wait(timeout=10)
print("完成!")
