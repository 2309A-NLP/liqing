"""
测试：分块模块
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
"""

import sys
from pathlib import Path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.chunker.text_splitter import Chunker


def test_chunk_simple(tc):
    """测试基本分块：300字 → 每块100字 → 至少2块"""
    text = "A" * 300
    chunker = Chunker(chunk_size=100, chunk_overlap=20)

    tc.info(f"  输入: 300字, chunk_size=100, overlap=20")
    pages = [{"text": text, "page_no": 1}]
    chunks = chunker.chunk_pages(pages, source_file="test.pdf")

    tc.assert_gt(len(chunks), 1, "分块数 > 1")
    tc.info(f"  实际分块: {len(chunks)} 块")
    for c in chunks:
        tc.assert_true("page_no" in c, "每块包含 page_no")
        tc.assert_true("source_file" in c, "每块包含 source_file")
        tc.assert_true("text" in c, "每块包含 text")
        tc.assert_true(len(c["text"]) <= 100, f"每块长度 ≤ 100 (实际={len(c['text'])})")


def test_chunk_multi_page(tc):
    """测试多页分块：2页 → 两页的块都出现"""
    chunker = Chunker(chunk_size=50, chunk_overlap=10)
    pages = [
        {"text": "第一页内容 " * 10, "page_no": 1},
        {"text": "第二页内容 " * 10, "page_no": 2},
    ]

    tc.info(f"  输入: 2页, 每页约50字, chunk_size=50")
    chunks = chunker.chunk_pages(pages, source_file="multi.pdf")

    pages_found = set(c["page_no"] for c in chunks)
    tc.assert_in(1, pages_found, "包含第1页的块")
    tc.assert_in(2, pages_found, "包含第2页的块")
    tc.info(f"  总块数: {len(chunks)}, 来源页: {sorted(pages_found)}")


def test_chunk_metadata(tc):
    """测试元数据保留：页码、文件名、块序号"""
    chunker = Chunker(chunk_size=20, chunk_overlap=5)
    chunks = chunker.chunk_text("这是测试文本内容", page_no=5, source_file="doc.pdf")

    for c in chunks:
        tc.assert_eq(c["page_no"], 5, f"块{c['chunk_index']} 页码=5")
        tc.assert_eq(c["source_file"], "doc.pdf", f"块{c['chunk_index']} 文件名=doc.pdf")
        tc.assert_true(c["chunk_index"] >= 0, f"块序号 ≥ 0 (实际={c['chunk_index']})")
    tc.info(f"  共 {len(chunks)} 块, 元数据全部保留")


def test_chunk_empty(tc):
    """测试空文本：空字符串 → 0块"""
    chunker = Chunker()
    chunks = chunker.chunk_text("", page_no=1)
    tc.assert_eq(len(chunks), 0, "空文本分块 = 0")
    tc.info("  ✓ 空文本不产生分块")
