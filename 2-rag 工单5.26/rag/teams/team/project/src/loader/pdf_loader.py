"""
PDF 文档加载器 — 使用 Fitz（pymupdf）解析
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
"""

import pymupdf
from typing import List, Dict, Any


class PDFLoader:
    """PDF 文档加载器，基于 Fitz 引擎"""

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.doc: pymupdf.Document | None = None

    def open(self) -> None:
        """打开 PDF 文件"""
        try:
            self.doc = pymupdf.open(self.file_path)
        except Exception as e:
            raise RuntimeError(f"无法打开 PDF 文件: {e}")

    def close(self) -> None:
        """关闭 PDF 文档"""
        if self.doc:
            self.doc.close()

    def extract_pages(self) -> List[Dict[str, Any]]:
        """提取所有页面的文字和表格

        Returns:
            [{"text": str, "page_no": int, "tables": List[dict]}, ...]
        """
        if not self.doc:
            self.open()

        pages = []
        for page_num, page in enumerate(self.doc):
            text = page.get_text("text")
            tables = self._extract_tables(page)
            pages.append({
                "text": text.strip(),
                "page_no": page_num + 1,  # 1-indexed
                "tables": tables,
            })
        return pages

    def _extract_tables(self, page) -> List[Dict[str, Any]]:
        """从页面提取表格数据"""
        tables = []
        try:
            found = page.find_tables()
            for table in found.tables:
                rows = table.extract()
                if rows:
                    tables.append({
                        "header": rows[0] if rows else [],
                        "rows": rows[1:] if len(rows) > 1 else [],
                        "bbox": table.bbox,
                    })
        except Exception:
            pass  # 表格解析非关键路径，失败则跳过
        return tables

    def extract_metadata(self) -> Dict[str, Any]:
        """提取文档元信息"""
        if not self.doc:
            self.open()
        meta = self.doc.metadata
        return {
            "title": meta.get("title", ""),
            "author": meta.get("author", ""),
            "subject": meta.get("subject", ""),
            "pages": len(self.doc),
            "file_path": self.file_path,
        }

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __del__(self):
        """对象销毁时关闭文档（防御性，避免重复 close）"""
        if self.doc is not None:
            try:
                self.doc.close()
            except (ValueError, RuntimeError):
                pass  # 文档已关闭则忽略
