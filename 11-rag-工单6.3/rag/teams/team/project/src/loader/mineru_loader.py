"""
MinerU 内容加载器 — 读取 content_list.json 并预处理
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统

MinerU (magic-pdf) 解析 PDF 后输出 content_list.json，每个元素是一个版面块：
  - type: text / header / table / image / page_number / seal / chart / equation
  - text: 文本内容（text/header 类型直接是字符串）
  - text_level: 标题层级（header 类型有 1/2/3...）
  - table_body / table_caption / table_footnote: 表格相关字段
  - page_idx: 页码（0-indexed）
  - bbox: 边界框坐标

本模块负责：
  1. 加载 JSON 并校验格式
  2. 过滤噪声块（page_number / seal / chart / equation / image）
  3. 合并相邻短文本块
  4. 表格 HTML → Markdown 转换（如有 table_body）
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional

# 噪声类型：分块时不保留
NOISE_TYPES = {"page_number", "seal", "chart", "equation", "image"}


def _to_str(val) -> str:
    """将 MinerU 字段统一转为 str（可能是 str 或 list）"""
    if isinstance(val, str):
        return val
    if isinstance(val, list):
        # list of str 或 list of dict
        parts = []
        for item in val:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(item.get("text", str(item)))
            else:
                parts.append(str(item))
        return " ".join(parts)
    return str(val) if val else ""


class MinerULoader:
    """MinerU content_list.json 加载器"""

    def __init__(self, content_list_path: str):
        self.path = Path(content_list_path)
        if not self.path.exists():
            raise FileNotFoundError(f"MinerU 输出文件不存在: {self.path}")

    def load(self) -> List[Dict[str, Any]]:
        """加载并预处理 content_list.json

        Returns:
            预处理后的 block 列表，每个 block 至少包含 type / text / page_idx
        """
        with open(self.path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        if not isinstance(raw, list):
            raise ValueError(f"content_list.json 格式错误：期望 list，实际 {type(raw).__name__}")

        # 第一步：过滤噪声 + 标准化字段
        blocks = []
        for item in raw:
            block_type = item.get("type", "text")
            if block_type in NOISE_TYPES:
                continue

            text = item.get("text", "")
            if isinstance(text, str):
                text = text.strip()
            else:
                text = str(text).strip()

            block = {
                "type": block_type,
                "text": text,
                "page_idx": item.get("page_idx", 0),
            }

            # 保留标题层级
            if block_type == "header":
                block["text_level"] = item.get("text_level", 1)

            # 保留表格字段（HTML 转换在 chunker 中完成）
            if block_type == "table":
                block["table_body"] = item.get("table_body", "")
                # caption/footnote 在 MinerU 中可能是 list，统一转 str
                block["table_caption"] = _to_str(item.get("table_caption", ""))
                block["table_footnote"] = _to_str(item.get("table_footnote", ""))

            blocks.append(block)

        # 第二步：合并相邻短文本块（同页、同类型）
        blocks = self._merge_short_blocks(blocks)

        return blocks

    def _merge_short_blocks(
        self,
        blocks: List[Dict[str, Any]],
        min_len: int = 30,
    ) -> List[Dict[str, Any]]:
        """合并相邻短文本块

        规则：
          - 只合并 text 类型的块
          - 相邻且同页的短块（< min_len 字）合并为一个块
          - header / table 类型不参与合并
        """
        if not blocks:
            return blocks

        merged = []
        buffer = None

        for block in blocks:
            # 非 text 类型：先 flush buffer，再直接保留
            if block["type"] != "text":
                if buffer:
                    merged.append(buffer)
                    buffer = None
                merged.append(block)
                continue

            text = block["text"]
            if not text:
                continue

            # 短块：尝试合并
            if len(text) < min_len:
                if buffer and buffer["page_idx"] == block["page_idx"]:
                    buffer["text"] += "\n" + text
                elif buffer:
                    merged.append(buffer)
                    buffer = {**block}
                else:
                    buffer = {**block}
                continue

            # 长块：flush buffer，保留当前块
            if buffer:
                merged.append(buffer)
                buffer = None
            merged.append(block)

        if buffer:
            merged.append(buffer)

        return merged


def find_content_list(output_dir: str, pdf_name: str) -> Optional[str]:
    """在 MinerU 输出目录中查找 content_list.json

    Args:
        output_dir: 文件所在目录（如 data/source_docs/）
        pdf_name: PDF 文件名（不含扩展名，如 "招股说明书2"）

    Returns:
        content_list.json 的路径，未找到返回 None
    """
    candidates = [
        Path(output_dir) / pdf_name / "auto" / f"{pdf_name}_content_list.json",
        Path(output_dir) / pdf_name / f"{pdf_name}_content_list.json",
        Path(output_dir) / f"{pdf_name}_content_list.json",
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    return None
