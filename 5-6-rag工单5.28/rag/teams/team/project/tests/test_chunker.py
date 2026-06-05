"""
测试：分块模块 — 基于 MinerU blocks
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
"""

import sys
from pathlib import Path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.chunker.text_splitter import Chunker


def test_chunk_text_basic(tc):
    """测试基本文本分块：300字 → 每块≤100字 → 至少2块"""
    text = "A" * 300
    chunker = Chunker(chunk_size=100, chunk_overlap=20)

    tc.info(f"  输入: 300字, chunk_size=100, overlap=20")
    chunks = chunker.chunk_text(text, page_no=1, source_file="test.pdf")

    tc.assert_gt(len(chunks), 1, "分块数 > 1")
    tc.info(f"  实际分块: {len(chunks)} 块")
    for c in chunks:
        tc.assert_true("page_no" in c, "每块包含 page_no")
        tc.assert_true("source_file" in c, "每块包含 source_file")
        tc.assert_true("text" in c, "每块包含 text")
        tc.assert_true(len(c["text"]) <= 100, f"每块长度 ≤ 100 (实际={len(c['text'])})")


def test_chunk_blocks_text(tc):
    """测试 MinerU text blocks 分块"""
    chunker = Chunker(chunk_size=100, chunk_overlap=20)
    blocks = [
        {"type": "header", "text": "第一章 总则", "page_idx": 0, "text_level": 1},
        {"type": "text", "text": "本招股说明书依据《证券法》等相关法律法规编制。" * 5, "page_idx": 0},
        {"type": "text", "text": "公司本次发行股票。" * 3, "page_idx": 1},
    ]

    tc.info(f"  输入: 2个text块 + 1个header块")
    chunks = chunker.chunk_blocks(blocks, source_file="test.pdf")

    tc.assert_gt(len(chunks), 0, "分块数 > 0")
    tc.info(f"  实际分块: {len(chunks)} 块")

    # 检查 section_path 被注入
    has_section = any("第一章 总则" in c.get("section_path", "") for c in chunks)
    tc.assert_true(has_section, "section_path 包含标题层级")

    # 检查 chunk_type
    for c in chunks:
        tc.assert_in(c["chunk_type"], ["text", "table"], "chunk_type 为 text 或 table")


def test_chunk_blocks_table(tc):
    """测试 MinerU table blocks 分块"""
    chunker = Chunker(chunk_size=500, chunk_overlap=50)
    blocks = [
        {"type": "header", "text": "财务数据", "page_idx": 5, "text_level": 2},
        {"type": "table", "text": "", "page_idx": 5,
         "table_body": "<table><tr><td>项目</td><td>2024年</td></tr><tr><td>收入</td><td>100万</td></tr></table>",
         "table_caption": "营业收入表", "table_footnote": "数据经审计"},
    ]

    tc.info(f"  输入: 1个table块（含HTML）")
    chunks = chunker.chunk_blocks(blocks, source_file="test.pdf")

    tc.assert_gt(len(chunks), 0, "分块数 > 0")
    table_chunks = [c for c in chunks if c["chunk_type"] == "table"]
    tc.assert_gt(len(table_chunks), 0, "至少1个table块")

    # 检查表格内容
    table_text = table_chunks[0]["text"]
    tc.assert_true("营业收入" in table_text or "项目" in table_text, "表格包含表头内容")
    tc.assert_true("财务数据" in table_chunks[0].get("section_path", ""), "section_path 包含上级标题")
    tc.info(f"  table文本预览: {table_text[:150]}")


def test_chunk_blocks_empty(tc):
    """测试空 blocks：空列表 → 0块"""
    chunker = Chunker()
    chunks = chunker.chunk_blocks([], source_file="empty.pdf")
    tc.assert_eq(len(chunks), 0, "空blocks → 0块")
    tc.info("  ✓ 空输入不产生分块")
