"""
RAGAS 对比评测脚本 — RAG vs LightRAG
工单12：LightRAG 优化任务

同时评测传统 RAG 和 LightRAG 两个通道，输出 RAGAS 指标对比报告。

用法：
  python eval_compare.py                    # 评测两个通道
  python eval_compare.py --rag-only         # 只评测传统 RAG
  python eval_compare.py --lightrag-only    # 只评测 LightRAG
  python eval_compare.py --mode local       # LightRAG 查询模式
"""

import argparse
import asyncio
import json
import os
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

import httpx

# ── 配置 ──
LLM_BASE_URL = os.environ.get("XIAOMI_BASE_URL", "https://token-plan-cn.xiaomimimo.com/v1")
LLM_API_KEY = os.environ.get("XIAOMI_API_KEY", "")
LLM_MODEL = "mimo-v2.5-pro"

# RAG API 地址
RAG_API_PORT = int(os.environ.get("RAG_API_PORT", "8004"))
RAG_API_URL = f"http://localhost:{RAG_API_PORT}/query"

# 测试题
TEST_QUESTIONS = [
    {"id": 5, "question": "武汉力源信息技术股份有限公司组织结构图中，销售部有几个部门构成，其中大客户销售部有几个销售处构成？"},
    {"id": 6, "question": "武汉力源信息技术股份有限公司招股意向书中，从 2008 年中国IC 市场应用结构与增长图中可以看出，增长率最快的是哪个行业？负增长的是哪个行业？"},
    {"id": 1, "question": "武汉力源信息技术股份有限公司本次发行股数是多少，占发行后总股本的比例是多少？"},
    {"id": 2, "question": "武汉力源信息技术股份有限公司本次募集资金拟投资哪些项目？"},
    {"id": 3, "question": "与武汉力源信息技术股份有限公司存在控制关系的关联方是谁，持股比例和本公司关系是什么？"},
    {"id": 4, "question": "与武汉力源信息技术股份有限公司不存在控制关系的关联方企业有哪些？"},
    {"id": 260, "question": "报告期内，武汉兴图新科电子股份有限公司来自军用领域的收入分别是多少？"},
]

logger = logging.getLogger("eval_compare")


def load_ground_truth(path: str = None) -> List[Dict[str, Any]]:
    """加载 ground truth，如果没有则使用测试题"""
    if path and Path(path).exists():
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            data = [{"id": k, **data[k]} for k in sorted(data.keys())]
        return data
    
    # 没有 ground_truth 文件，返回测试题（无标准答案）
    return TEST_QUESTIONS


def call_rag_api(question: str) -> Dict[str, Any]:
    """调用传统 RAG API"""
    payload = {"question": question, "session_id": f"eval_{int(time.time())}"}
    try:
        with httpx.Client(timeout=120) as client:
            resp = client.post(RAG_API_URL, json=payload)
            resp.raise_for_status()
            return resp.json()
    except Exception as e:
        return {"answer": f"ERROR: {e}", "sources": []}


async def call_lightrag(question: str, mode: str = "mix") -> Dict[str, Any]:
    """调用 LightRAG 查询"""
    from src.lightrag_channel import create_lightrag_instance, query_lightrag_with_contexts
    
    rag = await create_lightrag_instance()
    try:
        result = await query_lightrag_with_contexts(rag, question, mode=mode)
        return result
    finally:
        await rag.finalize_storages()


def get_ragas_llm():
    """配置 RAGAS 评测 LLM（MIMO）"""
    from langchain_openai import ChatOpenAI
    
    return ChatOpenAI(
        model=LLM_MODEL,
        api_key=LLM_API_KEY,
        base_url=LLM_BASE_URL,
        temperature=0,
        max_tokens=2048,
    )


def get_ragas_embeddings():
    """配置 RAGAS Embeddings（本地 bge-m3）"""
    from langchain_community.embeddings import HuggingFaceBgeEmbeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper
    
    _MODEL_DIR = os.environ.get("MODEL_DIR", r"D:\models")
    bge_path = os.environ.get("MODEL_BGE_M3_PATH", os.path.join(_MODEL_DIR, "bge-m3"))
    
    # WSL 路径转换
    if os.name != "nt" and ":" in bge_path:
        drive = bge_path[0].lower()
        rest = bge_path[2:].replace("\\", "/")
        bge_path = f"/mnt/{drive}{rest}"
    
    embeddings = HuggingFaceBgeEmbeddings(
        model_name=bge_path,
        model_kwargs={"device": "cpu"},  # 评测用 CPU 即可
        encode_kwargs={"normalize_embeddings": True},
    )
    return LangchainEmbeddingsWrapper(embeddings)


def run_ragas_evaluation(
    questions: List[str],
    answers: List[str],
    contexts_list: List[List[str]],
    ground_truths: List[str],
    channel_name: str,
) -> Dict[str, Any]:
    """执行 RAGAS 评测"""
    from ragas import evaluate
    from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
    from datasets import Dataset
    
    # 构建数据集
    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts_list,
        "ground_truth": ground_truths,
    })
    
    # 配置评测 LLM 和 Embeddings
    llm = get_ragas_llm()
    embeddings = get_ragas_embeddings()
    
    print(f"执行 {channel_name} RAGAS 评测...")
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=llm,
        embeddings=embeddings,
    )
    
    # 提取分数
    scores = result.scores
    summary = {}
    for metric_name in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        values = [s[metric_name] for s in scores if metric_name in s]
        if values:
            summary[metric_name] = sum(values) / len(values)
    
    return {
        "channel": channel_name,
        "summary": summary,
        "per_question": scores,
    }


def generate_compare_report(
    rag_result: Dict[str, Any] = None,
    lightrag_result: Dict[str, Any] = None,
    gt_data: List[Dict[str, Any]] = None,
    rag_answers: List[str] = None,
    lightrag_answers: List[str] = None,
) -> str:
    """生成对比报告（Markdown）"""
    lines = [
        "# RAGAS 对比评测报告",
        "",
        f"> 生成时间: {datetime.now().isoformat()}",
        f"> 评测 LLM: {LLM_MODEL}",
        f"> 评测框架: RAGAS",
        "",
        "---",
        "",
        "## 总览对比",
        "",
        "| 指标 | 传统 RAG | LightRAG | 差异 |",
        "|------|----------|----------|------|",
    ]
    
    metrics = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    metric_desc = {
        "faithfulness": "回答是否忠实于检索内容",
        "answer_relevancy": "回答是否切题",
        "context_precision": "检索内容是否相关",
        "context_recall": "检索是否覆盖答案所需信息",
    }
    
    for metric in metrics:
        rag_score = rag_result["summary"].get(metric, 0) if rag_result else 0
        lr_score = lightrag_result["summary"].get(metric, 0) if lightrag_result else 0
        diff = lr_score - rag_score
        diff_str = f"+{diff:.4f}" if diff > 0 else f"{diff:.4f}"
        lines.append(f"| {metric} | {rag_score:.4f} | {lr_score:.4f} | {diff_str} |")
    
    lines.extend(["", "---", "", "## 逐题对比", ""])
    
    if gt_data:
        for i, item in enumerate(gt_data):
            lines.append(f"### 题目 {item['id']}: {item['question']}")
            lines.append("")
            
            if rag_answers and i < len(rag_answers):
                lines.append(f"**传统 RAG 回答**: {rag_answers[i][:300]}")
                lines.append("")
            
            if lightrag_answers and i < len(lightrag_answers):
                lines.append(f"**LightRAG 回答**: {lightrag_answers[i][:300]}")
                lines.append("")
            
            # 逐题分数
            if rag_result and i < len(rag_result["per_question"]):
                rs = rag_result["per_question"][i]
                lines.append(f"传统 RAG 得分: faithfulness={rs.get('faithfulness',0):.4f} | relevancy={rs.get('answer_relevancy',0):.4f} | precision={rs.get('context_precision',0):.4f} | recall={rs.get('context_recall',0):.4f}")
            
            if lightrag_result and i < len(lightrag_result["per_question"]):
                ls = lightrag_result["per_question"][i]
                lines.append(f"LightRAG 得分: faithfulness={ls.get('faithfulness',0):.4f} | relevancy={ls.get('answer_relevancy',0):.4f} | precision={ls.get('context_precision',0):.4f} | recall={ls.get('context_recall',0):.4f}")
            
            lines.extend(["", "---", ""])
    
    return "\n".join(lines)


async def main():
    parser = argparse.ArgumentParser(description="RAGAS 对比评测")
    parser.add_argument("--rag-only", action="store_true", help="只评测传统 RAG")
    parser.add_argument("--lightrag-only", action="store_true", help="只评测 LightRAG")
    parser.add_argument("--mode", default="mix", help="LightRAG 查询模式")
    parser.add_argument("--ground-truth", default=None, help="ground_truth.json 路径")
    parser.add_argument("--port", type=int, default=RAG_API_PORT, help="RAG API 端口")
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
    
    # 加载测试题
    gt_data = load_ground_truth(args.ground_truth)
    questions = [item["question"] for item in gt_data]
    ground_truths = [item.get("ground_truth", "") for item in gt_data]
    
    print(f"📝 测试题数量: {len(gt_data)}")
    
    rag_result = None
    lightrag_result = None
    rag_answers = []
    lightrag_answers = []
    rag_contexts = []
    lightrag_contexts = []
    
    # ── 评测传统 RAG ──
    if not args.lightrag_only:
        print("\n" + "=" * 60)
        print("传统 RAG 通道评测")
        print("=" * 60)
        
        for item in gt_data:
            q = item["question"]
            print(f"  [{item['id']}] {q[:40]}...", end=" ", flush=True)
            result = call_rag_api(q)
            answer = result.get("answer", "")
            sources = result.get("sources", [])
            
            ctxs = []
            for s in sources[:3]:
                text = s.get("text_preview", "") or s.get("text", "")
                if text and len(text) > 20:
                    ctxs.append(text[:500])
            if not ctxs:
                ctxs = [answer[:500]]
            
            rag_answers.append(answer)
            rag_contexts.append(ctxs)
            print("ok")
            time.sleep(0.5)
        
        # RAGAS 评测
        if ground_truths and any(ground_truths):
            rag_result = run_ragas_evaluation(
                questions, rag_answers, rag_contexts, ground_truths, "传统 RAG"
            )
            print("\n传统 RAG 评测结果:")
            for k, v in rag_result["summary"].items():
                print(f"  {k}: {v:.4f}")
    
    # ── 评测 LightRAG ──
    if not args.rag_only:
        print("\n" + "=" * 60)
        print(f"LightRAG 通道评测 (mode={args.mode})")
        print("=" * 60)
        
        for item in gt_data:
            q = item["question"]
            print(f"  [{item['id']}] {q[:40]}...", end=" ", flush=True)
            result = await call_lightrag(q, mode=args.mode)
            
            lightrag_answers.append(result["answer"])
            lightrag_contexts.append(result["contexts"] or [result["answer"][:500]])
            print("ok")
            time.sleep(1)  # LightRAG 调用间隔长一点
        
        # RAGAS 评测
        if ground_truths and any(ground_truths):
            lightrag_result = run_ragas_evaluation(
                questions, lightrag_answers, lightrag_contexts, ground_truths, "LightRAG"
            )
            print("\nLightRAG 评测结果:")
            for k, v in lightrag_result["summary"].items():
                print(f"  {k}: {v:.4f}")
    
    # ── 生成报告 ──
    report_md = generate_compare_report(
        rag_result, lightrag_result, gt_data, rag_answers, lightrag_answers
    )
    
    # 保存报告
    output_dir = Path(__file__).resolve().parent.parent.parent / "artifacts" / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    md_path = output_dir / "ragas_compare_report.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(report_md)
    print(f"\n📄 对比报告: {md_path}")
    
    # 保存原始数据
    json_path = output_dir / "ragas_compare_result.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "rag_result": rag_result,
            "lightrag_result": lightrag_result,
            "rag_answers": rag_answers,
            "lightrag_answers": lightrag_answers,
        }, f, ensure_ascii=False, indent=2)
    print(f"📊 原始数据: {json_path}")


if __name__ == "__main__":
    asyncio.run(main())
