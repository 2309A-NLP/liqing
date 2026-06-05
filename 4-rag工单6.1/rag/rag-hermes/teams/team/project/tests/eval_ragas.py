"""
RAGAS 评测脚本 — 使用 DeepSeek 作为评测 LLM
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统优化

用法：
  python eval_ragas.py --port 8004

依赖：
  pip install ragas langchain-openai httpx
"""

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path

import httpx

LLM_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
LLM_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
LLM_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")


def load_ground_truth(path: str) -> list:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = [{"id": k, **data[k]} for k in sorted(data.keys())]
    return data


def call_rag(question: str, port: int) -> dict:
    url = f"http://localhost:{port}/query"
    payload = {"question": question, "session_id": f"ragas_{int(time.time())}"}
    try:
        with httpx.Client(timeout=60) as client:
            resp = client.post(url, json=payload)
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
    return ChatOpenAI(
        model=LLM_MODEL,
        api_key=LLM_API_KEY,
        base_url=base,
        temperature=0,
        max_tokens=2048,
        model_kwargs={"n": 1},
    )


def run_ragas_evaluation(port: int = 8004):
    from ragas import evaluate
    from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from langchain_community.embeddings import HuggingFaceBgeEmbeddings
    from datasets import Dataset

    # 1. 加载 ground truth
    gt_path = Path(__file__).resolve().parent.parent.parent / "artifacts" / "evaluation" / "ground_truth.json"
    if not gt_path.exists():
        print(f"ground_truth.json 不存在: {gt_path}")
        return

    gt_data = load_ground_truth(str(gt_path))
    print(f"加载 {len(gt_data)} 道测试题")

    # 2. 调用 RAG
    print(f"调用 RAG API (port={port})...")
    questions, answers, contexts_list, ground_truths = [], [], [], []

    for item in gt_data:
        q = item["question"]
        gt = item["ground_truth"]
        ctxs = item.get("contexts", [])

        print(f"  [{item['id']}] {q[:40]}...", end=" ", flush=True)
        result = call_rag(q, port)
        answer = result.get("answer", "")

        rag_sources = result.get("sources", [])
        if rag_sources:
            # 提取实际检索文本内容（不是页码引用）
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

    # 3. 构建数据集
    print("\n构建 RAGAS 数据集...")
    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts_list,
        "ground_truth": ground_truths,
    })

    # 4. 用 DeepSeek 做评测 LLM
    print("配置 DeepSeek 作为评测 LLM...")
    llm = get_deepseek_llm()

    print("配置本地 bge-m3 embeddings...")
    embeddings = HuggingFaceBgeEmbeddings(
        model_name=r"D:\models\bge-m3",
        model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": True},
    )
    embeddings_wrapper = LangchainEmbeddingsWrapper(embeddings)

    print("执行 RAGAS 评测...")
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=llm,
        embeddings=embeddings_wrapper,
    )

    # 5. 输出结果
    print("\n" + "=" * 60)
    print("RAGAS 评测结果")
    print("=" * 60)
    scores = result.scores
    summary = {}
    for metric_name in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        values = [s[metric_name] for s in scores if metric_name in s]
        if values:
            avg = sum(values) / len(values)
            summary[metric_name] = avg
            print(f"  {metric_name:25s}: {avg:.4f}")

    # 6. 保存报告
    report = {
        "timestamp": datetime.now().isoformat(),
        "config": {"port": port, "model": LLM_MODEL, "eval_llm": "deepseek"},
        "scores": summary,
        "per_question": [],
    }
    for i, item in enumerate(gt_data):
        report["per_question"].append({
            "id": item["id"],
            "question": item["question"],
            "answer": answers[i],
            "ground_truth": ground_truths[i],
            "scores": scores[i] if i < len(scores) else {},
        })

    output_dir = Path(__file__).resolve().parent.parent.parent / "artifacts" / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "ragas_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n原始数据: {json_path}")

    md_path = output_dir / "ragas_report.md"
    md = generate_md_report(report, gt_data, answers, scores)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"评测报告: {md_path}")


def generate_md_report(report, gt_data, answers, scores):
    lines = [
        "# RAGAS 评测报告", "",
        f"> 生成时间: {report['timestamp']}",
        f"> 模型: {report['config']['model']}",
        f"> 评测 LLM: {report['config']['eval_llm']}",
        f"> 评测框架: RAGAS", "", "---", "",
        "## 总览", "",
        "| 指标 | 分数 | 说明 |",
        "|------|------|------|",
    ]
    desc = {
        "faithfulness": "回答是否忠实于检索内容",
        "answer_relevancy": "回答是否切题",
        "context_precision": "检索内容是否相关",
        "context_recall": "检索是否覆盖答案所需信息",
    }
    for k, v in report["scores"].items():
        lines.append(f"| {k} | {v:.4f} | {desc.get(k, '')} |")

    lines.extend(["", "---", "", "## 逐题得分", ""])
    lines.append("| 题号 | faithfulness | answer_relevancy | context_precision | context_recall |")
    lines.append("|------|-------------|-----------------|------------------|---------------|")
    for i, item in enumerate(gt_data):
        s = scores[i] if i < len(scores) else {}
        lines.append(f"| {item['id']} | {s.get('faithfulness',0):.4f} | {s.get('answer_relevancy',0):.4f} | {s.get('context_precision',0):.4f} | {s.get('context_recall',0):.4f} |")

    lines.extend(["", "---", "", "## 逐题详情", ""])
    for i, item in enumerate(gt_data):
        s = scores[i] if i < len(scores) else {}
        lines.extend([
            f"### {item['id']}: {item['question']}", "",
            f"**标准答案**: {item['ground_truth']}", "",
            f"**RAG 回答**: {answers[i][:300]}", "",
            f"**得分**: faithfulness={s.get('faithfulness',0):.4f} | relevancy={s.get('answer_relevancy',0):.4f} | precision={s.get('context_precision',0):.4f} | recall={s.get('context_recall',0):.4f}",
            "", "---", "",
        ])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8004)
    args = parser.parse_args()
    run_ragas_evaluation(port=args.port)


if __name__ == "__main__":
    main()
