"""
Embedding 变体对比评测 — 原始 vs 微调 bge-base-zh-v1.5

用法:
  # 先启动 API 服务
  python run.py

  # 运行对比评测
  python tests/eval_compare.py

  # 指定端口
  python tests/eval_compare.py --port 8004

  # 只评测一个变体
  python tests/eval_compare.py --variants base_ft

  # 指定 ground truth 文件
  python tests/eval_compare.py --gt /path/to/ground_truth.json

依赖:
  pip install ragas langchain-openai httpx
"""

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path

import httpx

# 自动加载 ~/.hermes/.env
_env_path = Path.home() / ".hermes" / ".env"
if _env_path.exists():
    for line in _env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

# 评测 LLM
LLM_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
LLM_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
LLM_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")


def load_ground_truth(path: str) -> list:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = [{"id": k, **data[k]} for k in sorted(data.keys())]
    return data


def call_rag(question: str, port: int, variant: str = "m3") -> dict:
    """调用 RAG API，指定 variant"""
    import requests
    url = f"http://127.0.0.1:{port}/query"
    payload = {
        "question": question,
        "session_id": f"eval_{variant}_{int(time.time())}",
        "variant": variant,
    }
    try:
        resp = requests.post(url, json=payload, timeout=120)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        return {"answer": f"ERROR: {e}", "sources": []}


def get_deepseek_llm():
    """配置 DeepSeek 作为 RAGAS 评测 LLM"""
    from langchain_openai import ChatOpenAI

    base = LLM_BASE_URL.rstrip("/")
    if not base.endswith("/v1"):
        base += "/v1"

    class _PatchedChatOpenAI(ChatOpenAI):
        def generate(self, *args, **kwargs):
            if "n" in kwargs:
                kwargs["n"] = 1
            for msg_list in args:
                if isinstance(msg_list, list):
                    for msg in msg_list:
                        if hasattr(msg, "additional_kwargs") and "n" in msg.additional_kwargs:
                            msg.additional_kwargs["n"] = 1
            try:
                return super().generate(*args, **kwargs)
            except Exception:
                if "n" in kwargs:
                    kwargs["n"] = 1
                return super().generate(*args, **kwargs)

    return _PatchedChatOpenAI(
        model=LLM_MODEL,
        api_key=LLM_API_KEY,
        base_url=base,
        temperature=0,
        max_tokens=2048,
        n=1,
    )


def run_single_variant(variant: str, gt_data: list, port: int) -> dict:
    """对单个 variant 运行 RAGAS 评测

    Returns:
        {"variant": str, "scores": dict, "per_question": list}
    """
    from ragas import evaluate
    from ragas.metrics import Faithfulness, AnswerRelevancy, LLMContextPrecisionWithoutReference, LLMContextRecall
    from datasets import Dataset

    print(f"\n{'=' * 60}")
    print(f"评测变体: {variant}")
    print(f"{'=' * 60}")

    # 1. 调用 RAG
    questions, answers, contexts_list, ground_truths = [], [], [], []

    for item in gt_data:
        q = item["question"]
        gt = item["ground_truth"]
        ctxs = item.get("contexts", [])

        print(f"  [{item['id']}] {q[:50]}...", end=" ", flush=True)
        result = call_rag(q, port, variant)
        answer = result.get("answer", "")

        rag_sources = result.get("sources", [])
        if rag_sources:
            ctxs = []
            for s in rag_sources[:3]:
                text = s.get("text_preview", "") or s.get("text", "")
                if text and len(text) > 20:
                    ctxs.append(text[:500])
            if not ctxs:
                ctxs = [answer[:500]]
        elif not ctxs:
            ctxs = [answer[:500]]

        questions.append(q)
        answers.append(answer)
        contexts_list.append(ctxs)
        ground_truths.append(gt)
        print("ok")
        time.sleep(0.5)

    # 2. 构建数据集
    print(f"\n构建 RAGAS 数据集 ({variant})...")
    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts_list,
        "ground_truth": ground_truths,
    })

    # 3. 评测
    print(f"配置评测 LLM...")
    llm = get_deepseek_llm()

    print(f"配置本地 bge-m3 embeddings...")
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from langchain_community.embeddings import HuggingFaceBgeEmbeddings
    embeddings = HuggingFaceBgeEmbeddings(
        model_name=r"D:\models\bge-m3",
        model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": True},
    )
    embeddings_wrapper = LangchainEmbeddingsWrapper(embeddings)

    print(f"执行 RAGAS 评测 ({variant})...")
    result = evaluate(
        dataset,
        metrics=[Faithfulness(), AnswerRelevancy(), LLMContextPrecisionWithoutReference(), LLMContextRecall()],
        llm=llm,
        embeddings=embeddings_wrapper,
    )

    # 4. 收集结果
    scores = result.scores
    summary = {}
    for metric_name in scores[0].keys() if scores else []:
        values = [s[metric_name] for s in scores if metric_name in s]
        if values:
            summary[metric_name] = sum(values) / len(values)

    print(f"\n{variant} 评测结果:")
    for k, v in summary.items():
        print(f"  {k:35s}: {v:.4f}")

    per_question = []
    for i, item in enumerate(gt_data):
        per_question.append({
            "id": item["id"],
            "question": item["question"],
            "answer": answers[i],
            "ground_truth": ground_truths[i],
            "scores": scores[i] if i < len(scores) else {},
        })

    return {
        "variant": variant,
        "scores": summary,
        "per_question": per_question,
    }


def generate_comparison_report(results: list, output_dir: Path) -> None:
    """生成对比报告（JSON + Markdown）"""
    timestamp = datetime.now().isoformat()

    # JSON 报告
    json_report = {
        "timestamp": timestamp,
        "config": {"eval_llm": LLM_MODEL, "framework": "RAGAS"},
        "variants": [],
    }
    for r in results:
        json_report["variants"].append({
            "name": r["variant"],
            "scores": r["scores"],
            "per_question": r["per_question"],
        })

    json_path = output_dir / "eval_compare.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_report, f, ensure_ascii=False, indent=2)
    print(f"\nJSON 报告: {json_path}")

    # Markdown 报告
    md_lines = [
        "# Embedding 变体对比评测报告", "",
        f"> 生成时间: {timestamp}",
        f"> 评测 LLM: {LLM_MODEL}",
        f"> 评测框架: RAGAS", "",
        "---", "",
        "## 总览对比", "",
    ]

    # 对比表格
    if results:
        metric_names = list(results[0]["scores"].keys())
        header = "| 指标 | " + " | ".join(r["variant"] for r in results) + " | 变化 |"
        sep = "|------|" + "|".join(["---"] * len(results)) + "|------|"
        md_lines.extend([header, sep])

        for metric in metric_names:
            vals = [r["scores"].get(metric, 0) for r in results]
            row = f"| {metric} | " + " | ".join(f"{v:.4f}" for v in vals)
            if len(vals) == 2 and vals[0] > 0:
                diff = (vals[1] - vals[0]) / vals[0] * 100
                sign = "+" if diff > 0 else ""
                row += f" | {sign}{diff:.1f}% |"
            else:
                row += " | — |"
            md_lines.append(row)

    md_lines.extend(["", "---", "", "## 逐题对比", ""])

    # 逐题对比
    if results and len(results) >= 2:
        gt_data = results[0]["per_question"]
        metric_names = list(results[0]["scores"].keys())

        for i, item in enumerate(gt_data):
            md_lines.extend([f"### 题目 {item['id']}: {item['question']}", ""])
            md_lines.append(f"**标准答案**: {item['ground_truth']}")
            md_lines.append("")

            for r in results:
                q_data = r["per_question"][i] if i < len(r["per_question"]) else {}
                md_lines.append(f"**{r['variant']} 回答**: {q_data.get('answer', 'N/A')[:300]}")
                md_lines.append("")
                score_str = " | ".join(
                    f"{m}={q_data.get('scores', {}).get(m, 0):.4f}"
                    for m in metric_names
                )
                md_lines.append(f"得分: {score_str}")
                md_lines.append("")

            md_lines.extend(["---", ""])

    md_path = output_dir / "eval_compare.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
    print(f"Markdown 报告: {md_path}")


def main():
    parser = argparse.ArgumentParser(description="Embedding 变体对比评测")
    parser.add_argument("--port", type=int, default=8004, help="RAG API 端口")
    parser.add_argument("--variants", type=str, nargs="+", default=["base", "base_ft"],
                        help="要评测的变体列表 (default: base base_ft)")
    parser.add_argument("--gt", type=str, default=None,
                        help="ground_truth.json 路径（默认自动查找）")
    args = parser.parse_args()

    # 查找 ground_truth.json
    if args.gt:
        gt_path = Path(args.gt)
    else:
        candidates = [
            Path(__file__).resolve().parent.parent.parent.parent / "artifacts" / "evaluation" / "ground_truth.json",
            Path(__file__).resolve().parent.parent / "artifacts" / "evaluation" / "ground_truth.json",
        ]
        gt_path = None
        for p in candidates:
            if p.exists():
                gt_path = p
                break
        if not gt_path:
            print("❌ 未找到 ground_truth.json，请用 --gt 指定路径")
            return

    gt_data = load_ground_truth(str(gt_path))
    print(f"加载 {len(gt_data)} 道测试题: {gt_path}")

    # 检查 API 是否可用
    try:
        import requests
        resp = requests.get(f"http://127.0.0.1:{args.port}/health", timeout=10)
        if resp.status_code == 200:
            print(f"API 服务已连接 (port={args.port})")
    except Exception as e:
        print(f"❌ API 服务不可用 (port={args.port}): {e}")
        print("请先运行 python run.py 启动服务")
        return

    # 逐个变体评测
    results = []
    for variant in args.variants:
        r = run_single_variant(variant, gt_data, args.port)
        results.append(r)

    # 输出对比
    output_dir = Path(__file__).resolve().parent.parent.parent.parent / "artifacts" / "evaluation"
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


if __name__ == "__main__":
    main()
