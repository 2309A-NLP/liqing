"""
RAG 评测脚本 — RAG vs 纯 LLM 对比评测
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统

用法：
  python eval.py                        # 默认 3 次
  python eval.py --runs 1               # 快速跑 1 次
  python eval.py --port 8004            # 指定端口

输出：
  teams/team/artifacts/evaluation/report.md
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

import httpx

# ── 配置 ──
RAG_BASE = "http://localhost:{port}"
LLM_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
LLM_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
LLM_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")

# ── 10 道测试题（来自工单） ──
TEST_QUESTIONS: List[Dict[str, str]] = [
    {"id": "Q1", "question": "报告期内，武汉兴图新科电子股份有限公司来自军用领域的收入分别是多少？"},
    {"id": "Q2", "question": "武汉兴图新科电子股份有限公司参与制定了哪个技术标准？"},
    {"id": "Q3", "question": "报告期内，武汉兴图新科电子股份有限公司来自军用领域的收入占主营业务收入的比重分别是多少？"},
    {"id": "Q4", "question": "根据武汉兴图新科电子股份有限公司招股意向书，电子信息行业的上游涉及哪些企业？"},
    {"id": "Q5", "question": "武汉兴图新科电子股份有限公司在哪个领域已经成为重要供应商？"},
    {"id": "Q6", "question": "根据武汉兴图新科电子股份有限公司招股意向书，电子信息行业的下游主要包括哪些行业？"},
    {"id": "Q7", "question": "武汉兴图新科电子股份有限公司参与的哪个工程荣获了国家科技进步一等奖？"},
    {"id": "Q8", "question": "武汉兴图新科电子股份有限公司注册资本是多少？"},
    {"id": "Q9", "question": "武汉兴图新科电子股份有限公司法定代表人是谁？"},
    {"id": "Q10", "question": "武汉兴图新科电子股份有限公司计划使用本次发行募集资金的多少用于补充流动资金？"},
]


def call_rag(question: str, port: int = 8004) -> Dict[str, Any]:
    """调用 RAG /query 接口"""
    url = f"http://localhost:{port}/query"
    payload = {"question": question, "session_id": f"eval_{int(time.time())}"}
    t0 = time.perf_counter()
    try:
        with httpx.Client(timeout=60) as client:
            resp = client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
            latency = time.perf_counter() - t0
            return {
                "answer": data.get("answer", ""),
                "sources": data.get("sources", []),
                "latency_s": round(latency, 2),
                "status": "ok",
            }
    except Exception as e:
        return {"answer": "", "sources": [], "latency_s": 0, "status": f"error: {e}"}


def call_llm(question: str) -> Dict[str, Any]:
    """直接调用 LLM（无检索）"""
    if not LLM_API_KEY:
        return {"answer": "", "latency_s": 0, "status": "error: DEEPSEEK_API_KEY not set"}

    system_prompt = "你是一个专业的文档问答助手。请根据你的知识回答用户问题。如果不确定，请说明。使用中文回答。"
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]

    headers = {"Authorization": f"Bearer {LLM_API_KEY}", "Content-Type": "application/json"}
    base = LLM_BASE_URL.rstrip("/")
    if not base.endswith("/v1"):
        base += "/v1"
    url = f"{base}/chat/completions"
    payload = {"model": LLM_MODEL, "messages": messages, "temperature": 0.3, "max_tokens": 2048}

    t0 = time.perf_counter()
    try:
        with httpx.Client(timeout=60) as client:
            resp = client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
            latency = time.perf_counter() - t0
            answer = data["choices"][0]["message"]["content"]
            return {"answer": answer, "latency_s": round(latency, 2), "status": "ok"}
    except Exception as e:
        return {"answer": "", "latency_s": 0, "status": f"error: {e}"}


def run_evaluation(port: int = 8004, runs: int = 3) -> Dict[str, Any]:
    """执行完整评测"""
    results = []
    total_questions = len(TEST_QUESTIONS)
    total_runs = total_questions * runs * 2  # RAG + LLM each

    print(f"📊 开始评测: {total_questions} 题 × {runs} 次 × 2 模式 = {total_runs} 次调用")
    print(f"   RAG 端口: {port}")
    print(f"   LLM 模型: {LLM_MODEL}")
    print()

    for q in TEST_QUESTIONS:
        qid = q["id"]
        question = q["question"]
        print(f"[{qid}] {question[:50]}...")

        rag_results = []
        llm_results = []

        for run_idx in range(runs):
            # RAG
            print(f"  Run {run_idx+1}/{runs}: RAG...", end=" ", flush=True)
            rag = call_rag(question, port)
            rag_results.append(rag)
            print(f"{rag['latency_s']}s | {len(rag.get('sources', []))} sources | {rag['status']}")

            # LLM
            print(f"  Run {run_idx+1}/{runs}: LLM...", end=" ", flush=True)
            llm = call_llm(question)
            llm_results.append(llm)
            print(f"{llm['latency_s']}s | {llm['status']}")

            time.sleep(0.5)  # 避免请求过快

        # 汇总
        rag_avg_latency = round(sum(r["latency_s"] for r in rag_results) / runs, 2)
        llm_avg_latency = round(sum(r["latency_s"] for r in llm_results) / runs, 2)
        rag_sources_count = len(rag_results[0].get("sources", [])) if rag_results else 0

        results.append({
            "id": qid,
            "question": question,
            "rag_answer": rag_results[0]["answer"] if rag_results else "",
            "rag_sources": rag_results[0].get("sources", []) if rag_results else [],
            "rag_latency_avg": rag_avg_latency,
            "rag_status": rag_results[0]["status"] if rag_results else "no_run",
            "llm_answer": llm_results[0]["answer"] if llm_results else "",
            "llm_latency_avg": llm_avg_latency,
            "llm_status": llm_results[0]["status"] if llm_results else "no_run",
            "runs": runs,
        })
        print()

    return {
        "timestamp": datetime.now().isoformat(),
        "config": {"port": port, "runs": runs, "model": LLM_MODEL},
        "results": results,
    }


def generate_report(eval_data: Dict[str, Any]) -> str:
    """生成 Markdown 评测报告"""
    config = eval_data["config"]
    results = eval_data["results"]
    timestamp = eval_data["timestamp"]

    lines = [
        "# RAG 评测报告",
        "",
        f"> 生成时间: {timestamp}",
        f"> 测试轮次: 每题 {config['runs']} 次",
        f"> LLM 模型: {config['model']}",
        f"> RAG 端口: {config['port']}",
        "",
        "---",
        "",
        "## 总览",
        "",
        "| 指标 | 值 |",
        "|------|-----|",
    ]

    # 总览统计
    total = len(results)
    rag_ok = sum(1 for r in results if r["rag_status"] == "ok")
    llm_ok = sum(1 for r in results if r["llm_status"] == "ok")
    rag_avg = round(sum(r["rag_latency_avg"] for r in results) / total, 2) if total else 0
    llm_avg = round(sum(r["llm_latency_avg"] for r in results) / total, 2) if total else 0
    rag_sources_avg = round(sum(len(r["rag_sources"]) for r in results) / total, 1) if total else 0

    lines.append(f"| 测试题数 | {total} |")
    lines.append(f"| RAG 成功率 | {rag_ok}/{total} |")
    lines.append(f"| LLM 成功率 | {llm_ok}/{total} |")
    lines.append(f"| RAG 平均延迟 | {rag_avg}s |")
    lines.append(f"| LLM 平均延迟 | {llm_avg}s |")
    lines.append(f"| RAG 平均来源数 | {rag_sources_avg} |")
    lines.append(f"| 延迟差异 | RAG {'快' if rag_avg < llm_avg else '慢'} {abs(round(rag_avg - llm_avg, 2))}s |")
    lines.append("")
    lines.append("---")
    lines.append("")

    # 逐题对比
    lines.append("## 逐题对比")
    lines.append("")

    for r in results:
        lines.append(f"### {r['id']}: {r['question']}")
        lines.append("")

        # 来源
        if r["rag_sources"]:
            sources_str = ", ".join([f"第{s.get('page_no', '?')}页" for s in r["rag_sources"][:3]])
            lines.append(f"**RAG 来源**: {sources_str}")
            lines.append("")

        # RAG 回答
        lines.append(f"**RAG 回答** ({r['rag_latency_avg']}s):")
        lines.append(f"> {r['rag_answer'][:500]}")
        lines.append("")

        # LLM 回答
        lines.append(f"**纯 LLM 回答** ({r['llm_latency_avg']}s):")
        lines.append(f"> {r['llm_answer'][:500]}")
        lines.append("")

        lines.append("---")
        lines.append("")

    # 延迟对比表
    lines.append("## 延迟对比表")
    lines.append("")
    lines.append("| 题号 | RAG(s) | LLM(s) | 差异(s) | RAG 来源数 |")
    lines.append("|------|--------|--------|---------|-----------|")

    for r in results:
        diff = round(r["rag_latency_avg"] - r["llm_latency_avg"], 2)
        diff_str = f"+{diff}" if diff > 0 else str(diff)
        sources_count = len(r["rag_sources"])
        lines.append(f"| {r['id']} | {r['rag_latency_avg']} | {r['llm_latency_avg']} | {diff_str} | {sources_count} |")

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 结论")
    lines.append("")
    lines.append("（请手动填写：RAG 准确率、与纯 LLM 对比优势、待改进点）")
    lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="RAG 评测脚本")
    parser.add_argument("--port", type=int, default=8004, help="RAG 服务端口")
    parser.add_argument("--runs", type=int, default=3, help="每题测试轮次")
    parser.add_argument("--output", type=str, default="", help="输出路径")
    args = parser.parse_args()

    # 执行评测
    eval_data = run_evaluation(port=args.port, runs=args.runs)

    # 生成报告
    report = generate_report(eval_data)

    # 保存
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path(__file__).resolve().parent.parent.parent / "artifacts" / "evaluation" / "report.md"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    print(f"✅ 报告已保存: {output_path}")

    # 也保存原始 JSON
    json_path = output_path.with_suffix(".json")
    json_path.write_text(json.dumps(eval_data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✅ 原始数据: {json_path}")


if __name__ == "__main__":
    main()
