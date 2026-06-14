#!/usr/bin/env python3
"""工单15 测试脚本：验证图表引用检测和检索优化。

使用方式：
1. 确保RAGFlow后端已启动
2. 设置环境变量 RAGFLOW_API_URL（默认 http://localhost:9380）
3. 设置 CHAT_ID 和 API_TOKEN
4. python test_figure_retrieval.py
"""
import re
import sys
import os
import json

# ============================================================
# Part 1: detect_figure_reference 单元测试
# ============================================================

def detect_figure_reference(question: str) -> dict | None:
    """从query.py复制的函数，用于独立测试。"""
    if not question:
        return None
    result = {}
    m = re.search(r'第\s*(\d+)\s*页.*?图\s*(\d+)', question)
    if m:
        return {"page_num": int(m.group(1)), "figure_num": int(m.group(2))}
    m = re.search(r'图\s*(\d+).*?第\s*(\d+)\s*页', question)
    if m:
        return {"page_num": int(m.group(2)), "figure_num": int(m.group(1))}
    m = re.search(r'第\s*(\d+)\s*页', question)
    if m:
        result["page_num"] = int(m.group(1))
    m = re.search(r'图\s*(\d+)', question)
    if m:
        result["figure_num"] = int(m.group(1))
    if not result.get("figure_num"):
        m = re.search(r'(?:fig(?:ure)?\.?\s*(\d+))', question, re.IGNORECASE)
        if m:
            result["figure_num"] = int(m.group(1))
    return result if result else None


def run_unit_tests():
    """运行detect_figure_reference的单元测试。"""
    tests = [
        # 工单15的6个测试问题
        ("根据专利文本，本发明主要涉及哪种物料的分配装置？", None),
        ("根据专利文本，本发明的分散装置包含以下哪个组件？", None),
        ("在文件中第 11 页图 3 中，编号 13 的部件相对于编号 12 的部件的位置关系是？",
         {"page_num": 11, "figure_num": 3}),
        ("在文件中第 11 页图 3 中，编号 14 的部件位于整个装置的哪个位置？",
         {"page_num": 11, "figure_num": 3}),
        ("根据文件中第 11 页图 3，散料从部件 14 进入后，下一步会经过哪个部件？",
         {"page_num": 11, "figure_num": 3}),
        ("在文件中第 11 页图 3 的装置中，如果需要调整链条的位置，需要操作哪个部件?",
         {"page_num": 11, "figure_num": 3}),
        # 边界情况
        ("图3的结构是什么？", {"figure_num": 3}),
        ("第11页的内容是什么？", {"page_num": 11}),
        ("fig.3 shows what?", {"figure_num": 3}),
        ("figure 10 overview", {"figure_num": 10}),
        ("", None),
        ("这是一段普通文本", None),
        ("请看第 5 页的图 2 和图 3", {"page_num": 5, "figure_num": 2}),  # 匹配第一个
    ]

    passed = 0
    failed = 0
    for question, expected in tests:
        result = detect_figure_reference(question)
        if result == expected:
            passed += 1
            print(f"  PASS | {question[:50]}")
        else:
            failed += 1
            print(f"  FAIL | {question[:50]}")
            print(f"         got={result}, expected={expected}")

    print(f"\n单元测试结果: {passed} passed, {failed} failed, {passed + failed} total")
    return failed == 0


# ============================================================
# Part 2: 工单15的6个测试问题定义
# ============================================================

TEST_QUESTIONS = [
    {
        "id": 1,
        "question": "根据专利文本，本发明主要涉及哪种物料的分配装置？",
        "expected_answer": "块状散料",
        "type": "纯文本",
        "figure_ref": None,
    },
    {
        "id": 2,
        "question": "根据专利文本，本发明的分散装置包含以下哪个组件？",
        "expected_answer": "链条",
        "type": "纯文本",
        "figure_ref": None,
    },
    {
        "id": 3,
        "question": "在文件中第 11 页图 3 中，编号 13 的部件相对于编号 12 的部件的位置关系是？",
        "expected_answer": "位于编号 12 的部件之内",
        "type": "图文关联",
        "figure_ref": {"page_num": 11, "figure_num": 3},
    },
    {
        "id": 4,
        "question": "在文件中第 11 页图 3 中，编号 14 的部件位于整个装置的哪个位置？",
        "expected_answer": "顶部",
        "type": "图文关联",
        "figure_ref": {"page_num": 11, "figure_num": 3},
    },
    {
        "id": 5,
        "question": "根据文件中第 11 页图 3，散料从部件 14 进入后，下一步会经过哪个部件？",
        "expected_answer": "部件 13",
        "type": "图文关联",
        "figure_ref": {"page_num": 11, "figure_num": 3},
    },
    {
        "id": 6,
        "question": "在文件中第 11 页图 3 的装置中，如果需要调整链条的位置，需要操作哪个部件?",
        "expected_answer": "部件 11",
        "type": "图文关联",
        "figure_ref": {"page_num": 11, "figure_num": 3},
    },
]


def check_answer(answer: str, expected: str) -> bool:
    """检查答案是否包含期望的关键信息。"""
    if not answer or not expected:
        return False
    return expected.lower() in answer.lower()


# ============================================================
# Part 3: API集成测试（需要RAGFlow后端运行）
# ============================================================

def run_api_tests():
    """通过API运行6个测试问题。需要RAGFlow后端在运行。"""
    import urllib.request
    import urllib.error

    API_URL = sys.environ.get("RAGFLOW_API_URL", "http://localhost:9380")
    CHAT_ID = sys.environ.get("CHAT_ID", "")
    API_TOKEN = sys.environ.get("API_TOKEN", "")

    if not CHAT_ID or not API_TOKEN:
        print("跳过API测试：需要设置 CHAT_ID 和 API_TOKEN 环境变量")
        print("  export CHAT_ID=<your_chat_id>")
        print("  export API_TOKEN=<your_api_token>")
        return

    url = f"{API_URL}/api/v1/chats/{CHAT_ID}/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_TOKEN}",
    }

    results = []
    for tq in TEST_QUESTIONS:
        payload = json.dumps({
            "question": tq["question"],
            "stream": False,
        }).encode("utf-8")

        req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                answer = data.get("data", {}).get("answer", "")
                refs = data.get("data", {}).get("reference", {}).get("chunks", [])

                # Check answer correctness
                correct = check_answer(answer, tq["expected_answer"])

                # Check if figure chunks are in references (for figure questions)
                has_figure_ref = any(
                    ck.get("doc_type") == "image" or "[图表描述]" in (ck.get("content", "") or "")
                    for ck in refs
                )

                result = {
                    "id": tq["id"],
                    "type": tq["type"],
                    "question": tq["question"][:50],
                    "expected": tq["expected_answer"],
                    "answer_correct": correct,
                    "has_figure_chunk": has_figure_ref,
                    "ref_count": len(refs),
                    "answer_snippet": answer[:100] if answer else "(empty)",
                }
                results.append(result)

                status = "PASS" if correct else "FAIL"
                fig = " [有图chunk]" if has_figure_ref else " [无图chunk]"
                print(f"  Q{tq['id']} {status}{fig} | expected='{tq['expected_answer']}' | answer='{answer[:80]}'")

        except Exception as e:
            print(f"  Q{tq['id']} ERROR | {e}")
            results.append({"id": tq["id"], "error": str(e)})

    # Summary
    total = len(results)
    correct_count = sum(1 for r in results if r.get("answer_correct"))
    fig_questions = [r for r in results if r.get("type") == "图文关联"]
    fig_with_chunk = sum(1 for r in fig_questions if r.get("has_figure_chunk"))

    print(f"\n=== 测试总结 ===")
    print(f"总准确率: {correct_count}/{total} ({correct_count/total*100:.0f}%)")
    print(f"图文问题图chunk命中: {fig_with_chunk}/{len(fig_questions)}")

    # Write detailed results
    with open("test_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"详细结果已保存到 test_results.json")


if __name__ == "__main__":
    print("=" * 60)
    print("Part 1: detect_figure_reference 单元测试")
    print("=" * 60)
    unit_ok = run_unit_tests()

    print()
    print("=" * 60)
    print("Part 2: API集成测试 (需要RAGFlow后端)")
    print("=" * 60)
    run_api_tests()
