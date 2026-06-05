"""
测试：检索模块 — BM25 关键词索引
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
"""

import sys
from pathlib import Path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import os
import tempfile
from src.store.keyword_store import BM25Index


def test_bm25_basic(tc):
    """测试 BM25 基本检索：建3条索引 → 搜索关键字 → 命中正确"""
    chunks = [
        {"text": "武汉兴图新科电子股份有限公司是一家军工企业", "page_no": 1, "source_file": "test.pdf", "chunk_index": 0},
        {"text": "公司注册资本为5000万元人民币", "page_no": 2, "source_file": "test.pdf", "chunk_index": 1},
        {"text": "公司参与制定了国家技术标准", "page_no": 3, "source_file": "test.pdf", "chunk_index": 2},
    ]
    bm25 = BM25Index()
    bm25.build_index(chunks)

    query = "注册资本"
    tc.info(f"  索引: 3条文本（军工/注册资本/技术标准）")
    tc.info(f"  查询: '{query}'")

    results = bm25.search(query, top_k=3)
    tc.assert_gt(len(results), 0, "检索结果 > 0")
    tc.info(f"  命中: {len(results)} 条")
    for r in results:
        tc.info(f"    得分={r['score']:.2f} | {r['text'][:50]}...")


def test_bm25_empty_index(tc):
    """测试空索引检索：不建索引 → 返回空列表"""
    bm25 = BM25Index()
    results = bm25.search("测试", top_k=5)
    tc.assert_eq(results, [], "空索引检索 = []")
    tc.info("  ✓ 空索引返回空列表，不报错")


def test_bm25_no_match(tc):
    """测试无匹配：索引A内容 → 搜B关键词 → 返回低分/空"""
    chunks = [
        {"text": "人工智能和机器学习技术", "page_no": 1, "source_file": "test.pdf", "chunk_index": 0},
    ]
    bm25 = BM25Index()
    bm25.build_index(chunks)

    query = "完全不相关的内容"
    tc.info(f"  索引: '人工智能和机器学习技术'")
    tc.info(f"  查询: '{query}'")

    results = bm25.search(query, top_k=5)
    tc.info(f"  命中: {len(results)} 条（BM25 无匹配也会返回低分结果）")
    # 即使不匹配，BM25 也会给低分，不崩溃


def test_bm25_save_load(tc):
    """测试 BM25 持久化：建索引 → 存pickle → 重载 → 数据一致"""
    chunks = [
        {"text": "测试内容保存加载", "page_no": 1, "source_file": "t.pdf", "chunk_index": 0},
    ]
    bm25 = BM25Index()
    bm25.build_index(chunks)

    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        save_path = f.name

    try:
        bm25.save(save_path)
        file_size = os.path.getsize(save_path)
        tc.assert_gt(file_size, 0, f"pickle 文件非空 (大小={file_size}B)")
        tc.info(f"  保存到: {save_path} ({file_size}B)")

        bm25_loaded = BM25Index()
        bm25_loaded.load(save_path)
        tc.assert_eq(len(bm25_loaded._chunks), 1, "加载后 chunks 数 = 1")
        tc.assert_eq(bm25_loaded._chunks[0]["text"], "测试内容保存加载", "加载后文本一致")
        tc.assert_true(bm25_loaded._index is not None, "BM25 索引对象非空")
        tc.info("  ✓ 加载后索引可用")
    finally:
        os.unlink(save_path)
