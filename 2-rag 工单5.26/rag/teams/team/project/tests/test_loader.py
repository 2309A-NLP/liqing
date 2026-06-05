"""
测试：PDF 解析模块
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
"""

import sys
from pathlib import Path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import os
import tempfile
import pymupdf
from src.loader.pdf_loader import PDFLoader


def _create_dummy_pdf(path: str, text: str = "test content"):
    """创建一个简单测试 PDF"""
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), text, fontsize=12)
    doc.save(path)
    doc.close()
    return path


def test_extract_pages(tc):
    """测试基本文字提取：创建1页PDF → 解析 → 验证文字和页码"""
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        fname = f.name

    try:
        _create_dummy_pdf(fname, "Page 1 test content")

        tc.info(f"  输入: 1页PDF, 内容='Page 1 test content'")

        loader = PDFLoader(fname)
        pages = loader.extract_pages()
        loader.close()

        tc.assert_eq(len(pages), 1, "解析出1页")
        tc.assert_in("test content", pages[0]["text"], "提取的文字包含'test content'")
        tc.assert_eq(pages[0]["page_no"], 1, "页码为1")
    finally:
        os.unlink(fname)


def test_extract_metadata(tc):
    """测试元数据提取：验证页数和路径"""
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        fname = f.name

    try:
        _create_dummy_pdf(fname)
        loader = PDFLoader(fname)
        meta = loader.extract_metadata()
        loader.close()

        tc.assert_eq(meta["pages"], 1, "页数 = 1")
        tc.assert_eq(meta["file_path"], fname, "文件路径正确")
    finally:
        os.unlink(fname)


def test_multi_page(tc):
    """测试多页 PDF：创建3页 → 验证每页内容独立"""
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        fname = f.name

    try:
        doc = pymupdf.open()
        for i in range(3):
            page = doc.new_page()
            page.insert_text((72, 72), f"Page {i+1} content", fontsize=12)
        doc.save(fname)
        doc.close()

        tc.info("  输入: 3页PDF, 每页内容不同")

        loader = PDFLoader(fname)
        pages = loader.extract_pages()
        loader.close()

        tc.assert_eq(len(pages), 3, "解析出3页")
        for i, page in enumerate(pages):
            expected_text = f"Page {i+1}"
            tc.assert_in(expected_text, page["text"], f"第{i+1}页包含'{expected_text}'")
            tc.assert_eq(page["page_no"], i + 1, f"第{i+1}页页码={i+1}")
    finally:
        os.unlink(fname)


def test_missing_file(tc):
    """测试异常路径：读取不存在的文件 → 应抛出 RuntimeError"""
    try:
        loader = PDFLoader("/nonexistent/file.pdf")
        loader.open()
        tc.assert_true(False, "应抛出 RuntimeError")
    except RuntimeError as e:
        tc.info(f"  ✓ 正确抛出 RuntimeError: {str(e)[:80]}")


def test_context_manager(tc):
    """测试上下文管理器：with 语句自动关闭"""
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        fname = f.name

    try:
        _create_dummy_pdf(fname)
        with PDFLoader(fname) as loader:
            pages = loader.extract_pages()
            tc.assert_eq(len(pages), 1, "with 语句内解析正常")
        tc.info("  ✓ with 块退出后文档自动关闭")
    finally:
        os.unlink(fname)
