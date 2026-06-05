"""
测试：答案生成模块 — deepseek Prompt 构建
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
"""

import sys
from pathlib import Path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.generator.answer_gen import Generator
from src.config import config


def test_generate_basic(tc):
    """测试基本生成：传入检索结果 → 返回结构正确"""
    gen = Generator()
    context = [
        {"text": "武汉兴图新科电子股份有限公司注册资本为5000万元", "page_no": 2, "score": 0.95},
        {"text": "公司成立于2004年，主要从事军用视频通信", "page_no": 3, "score": 0.88},
    ]
    tc.info(f"  输入: 2条检索结果, 问题='公司注册资本是多少？'")
    tc.info(f"    第1条: page=2, score=0.95 (注册资本5000万)")
    tc.info(f"    第2条: page=3, score=0.88 (成立于2004年)")

    result = gen.generate(
        question="公司注册资本是多少？",
        context_chunks=context,
    )

    tc.assert_in("answer", result, "返回结果含 answer")
    tc.assert_in("sources", result, "返回结果含 sources")
    tc.assert_gt(len(result["sources"]), 0, "sources > 0")
    tc.info(f"  answer: {result['answer'][:100]}...")
    tc.info(f"  sources: {len(result['sources'])} 条")


def test_generate_with_history(tc):
    """测试带历史对话：传入上轮对话 → 上下文正常拼接"""
    gen = Generator()
    context = [
        {"text": "公司注册资本为5000万元人民币", "page_no": 2, "score": 0.95},
    ]
    history = [
        {"role": "user", "content": "介绍一下这家公司"},
        {"role": "assistant", "content": "武汉兴图新科电子股份有限公司是一家军工企业。"},
    ]
    tc.info(f"  历史: 2轮对话（介绍公司 → 回答）")
    tc.info(f"  问题: '注册资本呢？'（多轮追问）")

    result = gen.generate(
        question="注册资本呢？",
        context_chunks=context,
        history=history,
    )

    tc.assert_in("answer", result, "返回结果含 answer")
    tc.assert_gt(len(result["answer"]), 0, "回答非空")
    tc.info(f"  answer: {result['answer'][:100]}...")


def test_generate_empty_context(tc):
    """测试空上下文：没有检索结果 → 正常返回空 sources"""
    gen = Generator()
    result = gen.generate(
        question="测试问题",
        context_chunks=[],
    )

    tc.assert_in("answer", result, "返回结果含 answer")
    tc.assert_in("sources", result, "返回结果含 sources")
    tc.assert_eq(len(result["sources"]), 0, "sources = 0（未检索到内容）")
    tc.info(f"  answer: {result['answer'][:100]}...")
    tc.info("  ✓ 即使没有检索结果，也不崩溃，正常返回")


def test_generate_no_api_key(tc):
    """测试无 API Key：API Key 为空 → 返回友好错误提示"""
    old_key = config.DEEPSEEK_API_KEY
    config.DEEPSEEK_API_KEY = ""
    try:
        gen = Generator()
        context = [{"text": "测试内容", "page_no": 1, "score": 0.9}]
        tc.info(f"  条件: DEEPSEEK_API_KEY 为空")
        tc.info(f"  期望: 返回错误提示而非崩溃")

        result = gen.generate(
            question="测试",
            context_chunks=context,
        )

        tc.assert_in("answer", result, "返回结果含 answer")
        has_error = "异常" in result["answer"] or "失败" in result["answer"]
        tc.assert_true(has_error, f"回答包含错误提示: '{result['answer'][:80]}'")
        tc.info(f"  answer: {result['answer'][:100]}")
    finally:
        config.DEEPSEEK_API_KEY = old_key
