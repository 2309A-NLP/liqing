import json
import requests
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# ===================== 基础配置 =====================
RAG_API_URL = "http://127.0.0.1:8002/chat"
TEST_FILE = "test_questions.json"
OUTPUT_EXCEL = "RAG_完整测评_含检索上下文+回答.xlsx"
TOP_K = 5
# ====================================================

# 轻量中文嵌入模型，低显存
emb_model = SentenceTransformer("all-MiniLM-L6-v2")

# 读取测试集
with open(TEST_FILE, "r", encoding="utf-8") as f:
    test_data = json.load(f)

results = []

for idx, item in enumerate(test_data):
    question = item["question"]
    gt_answer = item.get("ground_truth", "")
    # 兼容：没有标准上下文也不报错
    gt_contexts = item.get("ground_truth_contexts", [])

    print(f"[{idx+1}/{len(test_data)}] 正在测评：{question}")

    # 调用你的RAG接口
    try:
        resp = requests.post(
            RAG_API_URL,
            json={
                "session_id": "eval_full",
                "user_id": "eval_full",
                "question": question,
                "role_name": "assistant"
            },
            timeout=30
        )
        data = resp.json()
        # RAG 生成的原始回答
        rag_answer = data.get("answer", "")
        # 检索召回的上下文列表
        retrieved_list = data.get("retrieved", [])
        # 拼接成完整文本，方便查看
        rag_contexts = [ctx.get("content", "") for ctx in retrieved_list]
        rag_contexts_str = "\n\n--------------------\n\n".join(rag_contexts)
    except Exception as e:
        print(f"接口异常跳过：{e}")
        results.append({
            "问题": question,
            "标准参考上下文": "",
            "RAG检索到的上下文": "接口请求失败",
            "标准回答": gt_answer,
            "RAG模型回答": "接口请求失败",
            "检索精确率Precision@K": 0,
            "检索召回率Recall@K": 0,
            "回答语义相似度": 0,
            "关键词重合度": 0,
            "综合质量得分": 0
        })
        continue

    # ========== 1. 检索指标：Precision / Recall ==========
    hit_count = 0
    for pred_ctx in rag_contexts:
        for gt_ctx in gt_contexts:
            emb_p = emb_model.encode(pred_ctx)
            emb_g = emb_model.encode(gt_ctx)
            sim = cosine_similarity([emb_p], [emb_g])[0][0]
            if sim >= 0.6:
                hit_count += 1
                break

    precision = hit_count / len(rag_contexts) if rag_contexts else 0
    recall = hit_count / len(gt_contexts) if gt_contexts else 0

    # ========== 2. 生成回答质量指标 ==========
    # 语义相似度
    emb_gt = emb_model.encode(gt_answer)
    emb_rag = emb_model.encode(rag_answer)
    sim_score = cosine_similarity([emb_gt], [emb_rag])[0][0]

    # 关键词重合度
    def get_word_set(text):
        text = text.replace("，", " ").replace("。", " ").replace("、", " ")
        return set(text.split())

    gt_words = get_word_set(gt_answer)
    rag_words = get_word_set(rag_answer)
    common_words = gt_words & rag_words
    key_score = len(common_words) / len(gt_words) if gt_words else 0

    # 综合得分
    final_score = 0.5 * sim_score + 0.3 * key_score + 0.2 * recall

    # ========== 存入完整数据（全部展示） ==========
    results.append({
        "问题": question,
        "标准参考上下文": "\n\n".join(gt_contexts),
        "RAG检索到的上下文": rag_contexts_str,
        "标准回答": gt_answer,
        "RAG模型回答": rag_answer,
        "检索精确率Precision@K": round(float(precision), 4),
        "检索召回率Recall@K": round(float(recall), 4),
        "回答语义相似度": round(float(sim_score), 4),
        "关键词重合度": round(float(key_score), 4),
        "综合质量得分": round(float(final_score), 4)
    })

# 汇总统计
df = pd.DataFrame(results)
avg_precision = df["检索精确率Precision@K"].mean()
avg_recall = df["检索召回率Recall@K"].mean()
avg_sim = df["回答语义相似度"].mean()
avg_key = df["关键词重合度"].mean()
avg_final = df["综合质量得分"].mean()

# 保存完整Excel（所有内容全保留）
with pd.ExcelWriter(OUTPUT_EXCEL, engine="openpyxl") as writer:
    df.to_excel(writer, index=False, sheet_name="详细测评数据")

# 打印汇总
print("\n" + "="*70)
print("📊 RAG 完整测评汇总（含检索上下文+模型回答）")
print("="*70)
print(f"平均检索精确率：{avg_precision:.4f}")
print(f"平均检索召回率：{avg_recall:.4f}")
print(f"平均语义相似度：{avg_sim:.4f}")
print(f"平均关键词重合：{avg_key:.4f}")
print(f"✅ RAG系统综合平均分：{avg_final:.4f}")
print("="*70)
print(f"完整报告已保存 → {OUTPUT_EXCEL}")