"""
LightRAG 通道 — 入库脚本
工单12：LightRAG 优化任务

读取 MinerU content_list.json，拼接文本，调用 LightRAG ainsert 入库。
LightRAG 会自动抽取实体/关系，构建知识图谱。

用法：
  python -m src.lightrag_channel.ingest                    # 入库所有文档
  python -m src.lightrag_channel.ingest --doc 1            # 只入库招股说明书1
  python -m src.lightrag_channel.ingest --doc 2            # 只入库招股说明书2
  python -m src.lightrag_channel.ingest --preview          # 只预览不入库
"""

import asyncio
import json
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any

from .config import SOURCE_DOCS_DIR

logger = logging.getLogger("lightrag_channel")


def load_mineru_blocks(content_list_path: str) -> List[Dict[str, Any]]:
    """加载 MinerU content_list.json，过滤噪声块"""
    with open(content_list_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    
    # 噪声类型
    NOISE_TYPES = {"page_number", "seal", "chart", "equation", "image"}
    
    blocks = []
    for item in raw:
        block_type = item.get("type", "text")
        if block_type in NOISE_TYPES:
            continue
        
        text = item.get("text", "").strip() if isinstance(item.get("text"), str) else str(item.get("text", "")).strip()
        
        block = {
            "type": block_type,
            "text": text,
            "page_idx": item.get("page_idx", 0),
        }
        
        # 保留表格字段
        if block_type == "table":
            block["table_body"] = item.get("table_body", "")
            block["table_caption"] = item.get("table_caption", "")
            if isinstance(block["table_caption"], list):
                block["table_caption"] = " ".join(str(x) for x in block["table_caption"])
        
        blocks.append(block)
    
    return blocks


def blocks_to_text(blocks: List[Dict[str, Any]], doc_name: str) -> str:
    """将 blocks 拼接成文本，供 LightRAG 入库
    
    策略：
    - text 块：直接拼接
    - header 块：加标题前缀
    - table 块：转 Markdown 格式
    """
    from bs4 import BeautifulSoup
    
    parts = []
    current_page = -1
    
    for block in blocks:
        block_type = block["type"]
        page_idx = block.get("page_idx", 0)
        
        # 换页标记（可选，帮助 LightRAG 理解文档结构）
        if page_idx != current_page:
            current_page = page_idx
            # 不加页码标记，避免噪声
        
        if block_type == "header":
            level = block.get("text_level", 1)
            text = block.get("text", "").strip()
            if text:
                # Markdown 标题格式
                parts.append(f"{'#' * level} {text}")
        
        elif block_type == "text":
            text = block.get("text", "").strip()
            if text and len(text) > 10:  # 过滤太短的文本
                parts.append(text)
        
        elif block_type == "table":
            table_body = block.get("table_body", "").strip()
            caption = block.get("table_caption", "").strip()
            
            if table_body:
                # 尝试 HTML → Markdown 转换
                markdown = _table_html_to_markdown(table_body)
                if markdown:
                    if caption:
                        parts.append(f"【表格】{caption}\n{markdown}")
                    else:
                        parts.append(markdown)
                else:
                    # 转换失败，保留原始 HTML 文本
                    soup = BeautifulSoup(table_body, "html.parser")
                    text = soup.get_text(separator=" ", strip=True)
                    if text:
                        if caption:
                            parts.append(f"【表格】{caption}\n{text}")
                        else:
                            parts.append(text)
    
    return "\n\n".join(parts)


def _table_html_to_markdown(html: str) -> str:
    """HTML 表格转 Markdown（简化版）"""
    try:
        from bs4 import BeautifulSoup
        
        soup = BeautifulSoup(html, "html.parser")
        table = soup.find("table")
        if not table:
            return ""
        
        rows = table.find_all("tr")
        if not rows:
            return ""
        
        md_lines = []
        for i, row in enumerate(rows):
            cells = row.find_all(["td", "th"])
            cell_texts = [c.get_text(strip=True).replace("\n", " ") for c in cells]
            md_lines.append("| " + " | ".join(cell_texts) + " |")
            
            # 第一行后加分隔线
            if i == 0:
                md_lines.append("| " + " | ".join(["---"] * len(cell_texts)) + " |")
        
        return "\n".join(md_lines)
    except Exception:
        return ""


def find_content_lists() -> Dict[str, str]:
    """查找所有 MinerU content_list.json
    
    Returns:
        {文档名: 文件路径}
    """
    result = {}
    source_dir = Path(SOURCE_DOCS_DIR)
    
    if not source_dir.exists():
        return result
    
    for p in source_dir.glob("*_content_list.json"):
        if "_v2" in p.name:
            continue
        doc_name = p.stem.replace("_content_list", "")
        result[doc_name] = str(p)
    
    return result


async def ingest_document(rag, content_list_path: str, doc_name: str, preview_only: bool = False):
    """入库单个文档
    
    Args:
        rag: LightRAG 实例
        content_list_path: content_list.json 路径
        doc_name: 文档名称
        preview_only: 只预览不入库
    """
    logger.info(f"📄 加载: {content_list_path}")
    blocks = load_mineru_blocks(content_list_path)
    logger.info(f"   有效块数: {len(blocks)}")
    
    # 拼接文本
    text = blocks_to_text(blocks, doc_name)
    logger.info(f"   文本长度: {len(text)} 字符")
    
    if preview_only:
        logger.info(f"   预览模式，不入库")
        # 保存预览文件
        preview_path = Path(SOURCE_DOCS_DIR).parent / "preview" / f"{doc_name}_lightrag_text.txt"
        preview_path.parent.mkdir(parents=True, exist_ok=True)
        with open(preview_path, "w", encoding="utf-8") as f:
            f.write(text[:5000])  # 只保存前5000字
        logger.info(f"   预览文件: {preview_path}")
        return
    
    # 入库
    logger.info(f"   开始入库...")
    await rag.ainsert(text)
    logger.info(f"   ✅ 入库完成: {doc_name}")


async def main():
    parser = argparse.ArgumentParser(description="LightRAG 入库脚本")
    parser.add_argument("--doc", type=int, choices=[1, 2], help="只入库指定文档（1或2）")
    parser.add_argument("--preview", action="store_true", help="只预览不入库")
    args = parser.parse_args()
    
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    
    # 查找文档
    content_lists = find_content_lists()
    if not content_lists:
        logger.error(f"❌ 未找到 content_list.json，请检查 {SOURCE_DOCS_DIR}")
        return
    
    # 按文档编号筛选
    if args.doc:
        keyword = f"招股说明书{args.doc}"
        filtered = {k: v for k, v in content_lists.items() if keyword in k}
        if not filtered:
            logger.error(f"❌ 未找到包含 '{keyword}' 的文档")
            return
        content_lists = filtered
    
    logger.info(f"📚 待处理文档: {len(content_lists)} 个")
    for name, path in content_lists.items():
        logger.info(f"   - {name}: {path}")
    
    # 初始化 LightRAG
    logger.info("🚀 初始化 LightRAG...")
    from .init_lightrag import create_lightrag_instance
    rag = await create_lightrag_instance()
    
    try:
        # 逐个入库
        for doc_name, content_list_path in content_lists.items():
            logger.info("")
            logger.info(f"{'#' * 60}")
            logger.info(f"# 文档: {doc_name}")
            logger.info(f"{'#' * 60}")
            
            await ingest_document(rag, content_list_path, doc_name, preview_only=args.preview)
        
        logger.info("")
        logger.info("=" * 60)
        if args.preview:
            logger.info("✅ 预览完成！去掉 --preview 参数即可正式入库。")
        else:
            logger.info("✅ 全部入库完成！")
        logger.info("=" * 60)
    
    finally:
        await rag.finalize_storages()


if __name__ == "__main__":
    asyncio.run(main())
