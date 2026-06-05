"""
文档分块 — 基于 MinerU content_list.json 的块级分块器
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统

分块策略：
  1. text 块：拼接 section_path 后按 chunk_size 切分（RecursiveCharacterTextSplitter）
  2. header 块：不单独成块，作为上下文注入后续 text/table 块
  3. table 块：每个表格生成两种 chunk：
     - table_semantic: Markdown 格式，适合语义搜索
     - table_json: JSON 格式，适合精确检索
     用 BeautifulSoup 解析 HTML，正确处理 rowspan/colspan 多级表头
"""

import sys
import types
import json
import os
from typing import List, Dict, Any, Tuple

# ── 屏蔽 TF/sentence_transformers 导入崩溃 ──
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

# Mock tf_keras（和 ingest.py 完全一致：用 _FakeModule，不用 MagicMock，不 mock keras）
class _FakeModule(types.ModuleType):
    """Mock 模块：属性访问返回空类，防 ImportError"""
    def __getattr__(self, name):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        return type(name, (), {"__init__": lambda self, *a, **k: None})

for _mod_name in ["tf_keras", "tf_keras.src", "tf_keras.src.losses", "tf_keras.src.activations"]:
    if _mod_name not in sys.modules:
        sys.modules[_mod_name] = _FakeModule(_mod_name)

# mock transformers.modeling_tf_utils（提供 TFPreTrainedModel 占位）
if "transformers.modeling_tf_utils" not in sys.modules:
    _tf_utils = _FakeModule("transformers.modeling_tf_utils")
    _tf_utils.TFPreTrainedModel = type("TFPreTrainedModel", (), {})
    sys.modules["transformers.modeling_tf_utils"] = _tf_utils

# 临时 mock sentence_transformers，让 langchain_text_splitters 导入时不触发 TF 崩溃链
_st_was_mocked = False
if "sentence_transformers" not in sys.modules:
    _st_mock = types.ModuleType("sentence_transformers")
    sys.modules["sentence_transformers"] = _st_mock
    _st_was_mocked = True

from langchain_text_splitters import RecursiveCharacterTextSplitter

# 清理所有 mock（让 embed.py/reranker.py 能导入真正的模块）
if _st_was_mocked and "sentence_transformers" in sys.modules:
    if sys.modules["sentence_transformers"] is _st_mock:
        del sys.modules["sentence_transformers"]

for _mod_name in ["tf_keras", "tf_keras.src", "tf_keras.src.losses", "tf_keras.src.activations"]:
    if _mod_name in sys.modules and isinstance(sys.modules[_mod_name], _FakeModule):
        del sys.modules[_mod_name]

if "transformers.modeling_tf_utils" in sys.modules and isinstance(sys.modules["transformers.modeling_tf_utils"], _FakeModule):
    del sys.modules["transformers.modeling_tf_utils"]


def table_html_to_markdown(html: str) -> str:
    """用 BeautifulSoup 将 HTML <table> 转 Markdown

    正确处理 rowspan/colspan，生成扁平化的单行表头。
    """
    from bs4 import BeautifulSoup

    if not html or "<table" not in html.lower():
        return ""

    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table")
    if not table:
        return ""

    rows = table.find_all("tr")
    if not rows:
        return ""

    # 构建网格（处理 rowspan/colspan）
    grid = []
    for row in rows:
        cells = row.find_all(["td", "th"])
        grid_row = []
        cell_idx = 0
        for cell in cells:
            # 跳过已被 rowspan/colspan 占据的位置
            while cell_idx < len(grid_row) and grid_row[cell_idx] is not None:
                cell_idx += 1

            text = cell.get_text(strip=True).replace("\n", " ")
            rs = int(cell.get("rowspan", 1))
            cs = int(cell.get("colspan", 1))

            # 填充当前单元格
            for _ in range(cs):
                while cell_idx < len(grid_row) and grid_row[cell_idx] is not None:
                    cell_idx += 1
                if cell_idx >= len(grid_row):
                    grid_row.append(text)
                else:
                    grid_row[cell_idx] = text
                cell_idx += 1

        grid.append(grid_row)

    # 规范化列数
    max_cols = max(len(r) for r in grid) if grid else 0
    for row in grid:
        while len(row) < max_cols:
            row.append("")

    if not grid or max_cols == 0:
        return ""

    # 生成 Markdown
    md_lines = []
    md_lines.append("| " + " | ".join(grid[0]) + " |")
    md_lines.append("| " + " | ".join(["---"] * max_cols) + " |")
    for row in grid[1:]:
        md_lines.append("| " + " | ".join(row) + " |")

    return "\n".join(md_lines)


def table_html_to_json(html: str) -> str:
    """用 BeautifulSoup 将 HTML <table> 转 JSON

    处理 rowspan/colspan，扁平化多级表头为 "A-B-C" 格式。
    返回 JSON 字符串（数组 of 对象）。
    """
    from bs4 import BeautifulSoup

    if not html or "<table" not in html.lower():
        return ""

    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table")
    if not table:
        return ""

    rows = table.find_all("tr")
    if len(rows) < 2:
        return ""

    # 解析所有行，处理 rowspan/colspan
    parsed_rows = []
    span_tracker = {}  # (row_idx, col_idx) -> text (from rowspan)

    for row_idx, row in enumerate(rows):
        cells = row.find_all(["td", "th"])
        parsed_row = []
        col_idx = 0

        for cell in cells:
            # 跳过被 rowspan 占据的位置
            while (row_idx, col_idx) in span_tracker:
                parsed_row.append(span_tracker[(row_idx, col_idx)])
                col_idx += 1

            text = cell.get_text(strip=True).replace("\n", " ")
            rs = int(cell.get("rowspan", 1))
            cs = int(cell.get("colspan", 1))

            # 填充当前单元格 + colspan
            for c in range(cs):
                while (row_idx, col_idx) in span_tracker:
                    parsed_row.append(span_tracker[(row_idx, col_idx)])
                    col_idx += 1
                parsed_row.append(text)
                # 标记 rowspan
                if rs > 1:
                    for r in range(1, rs):
                        span_tracker[(row_idx + r, col_idx)] = text
                col_idx += 1

        # 补齐行尾被 rowspan 占据的单元格
        while (row_idx, col_idx) in span_tracker:
            parsed_row.append(span_tracker[(row_idx, col_idx)])
            col_idx += 1

        parsed_rows.append(parsed_row)

    if len(parsed_rows) < 2:
        return ""

    # 规范化列数
    max_cols = max(len(r) for r in parsed_rows)
    for row in parsed_rows:
        while len(row) < max_cols:
            row.append("")

    # 第一行作为表头，扁平化多级表头
    headers = parsed_rows[0]

    # 数据行 → JSON 对象
    records = []
    for row in parsed_rows[1:]:
        record = {}
        for i, val in enumerate(row):
            key = headers[i] if i < len(headers) else f"col_{i}"
            if not key:
                key = f"col_{i}"
            record[key] = val
        records.append(record)

    return json.dumps(records, ensure_ascii=False, indent=2)


class Chunker:
    """基于 MinerU blocks 的文档分块器

    输入：mineru_loader.load() 输出的 block 列表
    输出：chunk 列表，每个 chunk 包含 text / page_no / source_file / chunk_index / chunk_type
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 128,
        min_chunk_len: int = 30,
    ):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", "。", ".", " ", ""],
            length_function=len,
        )
        self.min_chunk_len = min_chunk_len

    def chunk_blocks(
        self,
        blocks: List[Dict[str, Any]],
        source_file: str = "",
    ) -> List[Dict[str, Any]]:
        """将 MinerU blocks 切分为 RAG chunks

        Args:
            blocks: MinerULoader.load() 的输出
            source_file: 源文件名

        Returns:
            chunk 列表
        """
        chunks = []
        chunk_index = 0

        # 标题层级栈：{1: "第一节", 2: "一、概述", ...}
        header_stack: Dict[int, str] = {}

        # 短文本合并缓冲区
        text_buffer = ""
        buffer_page_idx = 0
        buffer_has_content = False

        for block in blocks:
            block_type = block["type"]
            page_idx = block["page_idx"]
            page_no = page_idx + 1  # 转为 1-indexed

            # ── 标题块：更新层级栈，不单独成块 ──
            if block_type == "header":
                # 先 flush 缓冲区
                if text_buffer.strip():
                    idx = self._flush_text_buffer(
                        text_buffer, buffer_page_idx, source_file,
                        header_stack, chunks, chunk_index,
                    )
                    chunk_index = idx
                    text_buffer = ""
                    buffer_has_content = False

                level = block.get("text_level", 1)
                text = block.get("text", "").strip()
                if text:
                    header_stack[level] = text
                    # 清除更深层级
                    for k in list(header_stack.keys()):
                        if k > level:
                            del header_stack[k]
                continue

            # ── 表格块：生成 table_semantic + table_json ──
            if block_type == "table":
                # 先 flush 缓冲区
                if text_buffer.strip():
                    idx = self._flush_text_buffer(
                        text_buffer, buffer_page_idx, source_file,
                        header_stack, chunks, chunk_index,
                    )
                    chunk_index = idx
                    text_buffer = ""
                    buffer_has_content = False

                section_path = self._get_section_path(header_stack)
                table_body = block.get("table_body", "").strip()
                caption = block.get("table_caption", "").strip()
                footnote = block.get("table_footnote", "").strip()

                if not table_body:
                    continue

                # ① table_semantic: Markdown 格式
                markdown = table_html_to_markdown(table_body)
                if markdown and len(markdown) > 10:
                    semantic_parts = []
                    if caption:
                        semantic_parts.append(f"【表格标题】{caption}")
                    semantic_parts.append(markdown)
                    if footnote:
                        semantic_parts.append(f"【脚注】{footnote}")
                    semantic_text = "\n".join(semantic_parts)

                    # 长表格切分
                    if len(semantic_text) > 1500:
                        sub_chunks = self.splitter.split_text(semantic_text)
                        for sub_text in sub_chunks:
                            chunks.append({
                                "text": sub_text,
                                "page_no": page_no,
                                "source_file": source_file,
                                "chunk_index": chunk_index,
                                "chunk_type": "table_semantic",
                                "section_path": section_path,
                            })
                            chunk_index += 1
                    else:
                        chunks.append({
                            "text": semantic_text,
                            "page_no": page_no,
                            "source_file": source_file,
                            "chunk_index": chunk_index,
                            "chunk_type": "table_semantic",
                            "section_path": section_path,
                        })
                        chunk_index += 1

                # ② table_json: JSON 格式
                json_str = table_html_to_json(table_body)
                if json_str and len(json_str) > 20:
                    json_parts = []
                    if caption:
                        json_parts.append(f"【表格标题】{caption}")
                    json_parts.append(json_str)
                    if footnote:
                        json_parts.append(f"【脚注】{footnote}")
                    json_text = "\n".join(json_parts)

                    if len(json_text) > 1500:
                        sub_chunks = self.splitter.split_text(json_text)
                        for sub_text in sub_chunks:
                            chunks.append({
                                "text": sub_text,
                                "page_no": page_no,
                                "source_file": source_file,
                                "chunk_index": chunk_index,
                                "chunk_type": "table_json",
                                "section_path": section_path,
                            })
                            chunk_index += 1
                    else:
                        chunks.append({
                            "text": json_text,
                            "page_no": page_no,
                            "source_file": source_file,
                            "chunk_index": chunk_index,
                            "chunk_type": "table_json",
                            "section_path": section_path,
                        })
                        chunk_index += 1

                continue

            # ── 文本块：合并短块，长块直接分块 ──
            if block_type == "text":
                text = block.get("text", "").strip()
                if not text:
                    continue

                # 短块累积到缓冲区
                if len(text) < 60:
                    if buffer_has_content and buffer_page_idx == page_idx:
                        text_buffer += "\n" + text
                    else:
                        # 跨页或缓冲区为空：先 flush 旧的
                        if text_buffer.strip():
                            idx = self._flush_text_buffer(
                                text_buffer, buffer_page_idx, source_file,
                                header_stack, chunks, chunk_index,
                            )
                            chunk_index = idx
                        text_buffer = text
                        buffer_page_idx = page_idx
                    buffer_has_content = True
                    continue

                # 长块：flush 缓冲区 + 处理当前块
                if text_buffer.strip():
                    idx = self._flush_text_buffer(
                        text_buffer, buffer_page_idx, source_file,
                        header_stack, chunks, chunk_index,
                    )
                    chunk_index = idx
                    text_buffer = ""
                    buffer_has_content = False

                section_path = self._get_section_path(header_stack)
                prefixed = self._prefix_with_section(text, section_path)
                sub_chunks = self.splitter.split_text(prefixed)

                for sub_text in sub_chunks:
                    if len(sub_text.strip()) < self.min_chunk_len:
                        continue
                    chunks.append({
                        "text": sub_text,
                        "page_no": page_no,
                        "source_file": source_file,
                        "chunk_index": chunk_index,
                        "chunk_type": "text",
                        "section_path": section_path,
                    })
                    chunk_index += 1

        # flush 最后的缓冲区
        if text_buffer.strip():
            chunk_index = self._flush_text_buffer(
                text_buffer, buffer_page_idx, source_file,
                header_stack, chunks, chunk_index,
            )

        return chunks

    def _flush_text_buffer(
        self,
        buffer: str,
        page_idx: int,
        source_file: str,
        header_stack: Dict[int, str],
        chunks: List[Dict[str, Any]],
        chunk_index: int,
    ) -> int:
        """将文本缓冲区的内容分块并追加到 chunks"""
        page_no = page_idx + 1
        section_path = self._get_section_path(header_stack)
        prefixed = self._prefix_with_section(buffer.strip(), section_path)
        sub_chunks = self.splitter.split_text(prefixed)

        for sub_text in sub_chunks:
            # 过滤过短的 chunk（如单独的"。"、"[招股意向书]"等无信息量片段）
            if len(sub_text.strip()) < self.min_chunk_len:
                continue
            chunks.append({
                "text": sub_text,
                "page_no": page_no,
                "source_file": source_file,
                "chunk_index": chunk_index,
                "chunk_type": "text",
                "section_path": section_path,
            })
            chunk_index += 1

        return chunk_index

    @staticmethod
    def _get_section_path(header_stack: Dict[int, str]) -> str:
        """从标题栈构建章节路径"""
        if not header_stack:
            return ""
        sorted_levels = sorted(header_stack.keys())
        return " > ".join(header_stack[k] for k in sorted_levels)

    @staticmethod
    def _prefix_with_section(text: str, section_path: str) -> str:
        """给文本添加章节路径前缀"""
        if not section_path:
            return text
        if len(section_path) < 100:
            return f"[{section_path}]\n{text}"
        return text

    # ── 兼容旧接口（测试用）──

    def chunk_text(
        self,
        text: str,
        page_no: int = 0,
        source_file: str = "",
    ) -> List[Dict[str, Any]]:
        """直接对文本分块（用于测试）"""
        chunks = []
        split_docs = self.splitter.create_documents([text])
        for i, doc in enumerate(split_docs):
            chunks.append({
                "text": doc.page_content,
                "page_no": page_no,
                "source_file": source_file,
                "chunk_index": i,
                "chunk_type": "text",
                "section_path": "",
            })
        return chunks
