"""
PDF 离线入库脚本 — 基于 MinerU 解析结果，支持多文档批量入库
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统

流程：MinerU content_list.json → 分块 → 向量化 → Milvus 入库

支持多文档：
  - 招股说明书1（兴图新科）
  - 招股说明书2（力源信息）
  - 后续新增文档放到 data/source_docs/ 目录即可自动发现

用法：
  python ingest.py                                            # 自动发现所有 content_list.json 全部入库
  python ingest.py --content-list a.json b.json               # 指定一个或多个文件
  python ingest.py --content-list a.json --preview             # 只预览不入库
  python ingest.py --rebuild                                   # 清空 Milvus 后重新入库全部文档
  python ingest.py --no-bm25                                   # 跳过 BM25 索引
"""

import sys
import json
import argparse
from pathlib import Path
from typing import List

# 禁用 TensorFlow + 噪音
import os
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

# 路径修复
_PROJECT_ROOT = Path(__file__).parent.resolve()
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.logger import logger


def find_all_content_lists() -> List[str]:
    """在 data/source_docs/ 目录中查找所有 content_list.json

    返回按文件名排序的路径列表
    """
    from src.config import config
    source_dir = config.SOURCE_DOCS_DIR

    if not source_dir.exists():
        return []

    candidates = [
        str(p) for p in source_dir.glob("*_content_list.json")
        if "_v2" not in p.name
    ]

    # 按文件名排序，保证顺序一致
    candidates.sort(key=lambda p: Path(p).name)
    return candidates


def save_preview(
    content_list_path: str,
    blocks: list,
    chunks: list,
) -> None:
    """保存解析和分块的预览文件

    每个文档独立保存，文件名带文档标识：
      data/preview/{文档名}_blocks.json
      data/preview/{文档名}_chunks.json
    """
    preview_dir = Path(_PROJECT_ROOT) / "data" / "preview"
    preview_dir.mkdir(parents=True, exist_ok=True)

    doc_name = Path(content_list_path).stem.replace("_content_list", "")

    # 1. blocks 预览
    blocks_preview = []
    for b in blocks:
        blocks_preview.append({
            "type": b["type"],
            "page_idx": b["page_idx"],
            "text_length": len(b.get("text", "")),
            "text_preview": b.get("text", "")[:200],
            "text_level": b.get("text_level"),
            "has_table_markdown": bool(b.get("table_markdown")),
        })

    blocks_path = preview_dir / f"{doc_name}_blocks.json"
    with open(blocks_path, "w", encoding="utf-8") as f:
        json.dump(blocks_preview, f, ensure_ascii=False, indent=2)
    logger.info(f"📄 blocks 预览已保存: {blocks_path}")

    # 2. chunks 预览
    chunks_preview = []
    for c in chunks:
        chunks_preview.append({
            "chunk_type": c["chunk_type"],
            "page_no": c["page_no"],
            "text_length": len(c["text"]),
            "text_preview": c["text"][:300] + "..." if len(c["text"]) > 300 else c["text"],
            "section_path": c.get("section_path", ""),
        })

    chunks_path = preview_dir / f"{doc_name}_chunks.json"
    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(chunks_preview, f, ensure_ascii=False, indent=2)
    logger.info(f"📦 chunks 预览已保存: {chunks_path}")

    # 3. 统计摘要
    from collections import Counter
    block_types = Counter(b["type"] for b in blocks)
    chunk_types = Counter(c["chunk_type"] for c in chunks)
    page_range = set(b["page_idx"] for b in blocks)

    logger.info("")
    logger.info(f"{'=' * 50}")
    logger.info(f"📊 解析统计 — {Path(content_list_path).name}")
    logger.info(f"{'=' * 50}")
    logger.info(f"  源文件:      {content_list_path}")
    logger.info(f"  有效blocks:  {len(blocks)} (来自 {len(page_range)} 页)")
    logger.info(f"  block类型:   {dict(block_types)}")
    logger.info(f"{'=' * 50}")
    logger.info(f"📦 分块统计")
    logger.info(f"{'=' * 50}")
    logger.info(f"  文本块:      {chunk_types.get('text', 0)} 条")
    logger.info(f"  表格语义块:  {chunk_types.get('table_semantic', 0)} 条")
    logger.info(f"  表格JSON块:  {chunk_types.get('table_json', 0)} 条")
    logger.info(f"  总计:        {len(chunks)} 条")
    logger.info(f"{'=' * 50}")
    logger.info(f"📁 预览文件:")
    logger.info(f"  {blocks_path}")
    logger.info(f"  {chunks_path}")


def process_one_document(
    content_list_path: str,
    preview_only: bool = False,
    no_bm25: bool = False,
    embedder=None,
) -> List[dict]:
    """处理单个文档：加载 → 分块 → 向量化 → 入库

    Args:
        content_list_path: content_list.json 路径
        preview_only: 只预览不入库
        no_bm25: 跳过 BM25 索引
        embedder: 复用的 Embedder 实例（避免重复加载模型）

    Returns:
        该文档生成的 chunks 列表（给 BM25 用）
    """
    from src.loader.mineru_loader import MinerULoader
    from src.chunker.text_splitter import Chunker

    doc_name = Path(content_list_path).stem.replace("_content_list", "")

    # 1. 加载
    logger.info(f"📄 加载: {content_list_path}")
    loader = MinerULoader(content_list_path)
    blocks = loader.load()
    logger.info(f"   加载完成: {len(blocks)} 个有效块")

    # 2. 分块
    chunker = Chunker()
    chunks = chunker.chunk_blocks(blocks, source_file=doc_name)
    logger.info(f"   分块完成: {len(chunks)} 块")

    if not chunks:
        logger.error(f"❌ {doc_name}: 解析成功但未生成有效分块")
        return []

    # 3. 保存预览
    save_preview(content_list_path, blocks, chunks)

    if preview_only:
        return chunks

    # 4. 向量化
    if embedder is None:
        from src.embedder.embed import Embedder
        embedder = Embedder()

    texts = [c["text"] for c in chunks]
    logger.info(f"🧠 向量化中...（共 {len(texts)} 条）")
    embeddings = embedder.embed_batch(texts)
    logger.info(f"   向量化完成: {len(embeddings)} 条 (dim={len(embeddings[0])})")

    # 5. Milvus 入库
    from src.store.milvus_store import MilvusStore
    from src.config import config as _cfg2
    logger.info(f"🗄️  Milvus 入库中... (集合: {_cfg2.MILVUS_COLLECTION})")
    milvus = MilvusStore(collection_name=_cfg2.MILVUS_COLLECTION)
    milvus.connect()
    inserted = milvus.insert_chunks(chunks, embeddings)
    total = milvus.count()
    milvus.close()
    logger.info(f"   入库完成: {inserted} 条 (总计: {total})")

    return chunks


def _run_all_variants(args):
    """一次性入库所有变体：m3 → base → base_ft，各自独立集合"""
    from src.config import config as _cfg

    variants = ["m3", "base", "base_ft"]
    content_lists = args.content_list
    if not content_lists:
        content_lists = find_all_content_lists()
    if not content_lists:
        logger.error("❌ 未找到任何 MinerU content_list.json")
        sys.exit(1)

    logger.info(f"{'#' * 60}")
    logger.info(f"🚀 全变体入库模式 — {len(variants)} 个变体 × {len(content_lists)} 个文档")
    logger.info(f"{'#' * 60}")

    for vi, variant in enumerate(variants, 1):
        vc = _cfg.get_variant_config(variant)
        _cfg.EMBED_VARIANT = variant
        _cfg.EMBEDDING_DIM = vc["dim"]
        _cfg.MILVUS_COLLECTION = vc["collection"]

        logger.info(f"\n{'=' * 60}")
        logger.info(f"🔀 变体 {vi}/{len(variants)}: {variant}")
        logger.info(f"   模型: {vc['model_path']}")
        logger.info(f"   集合: {vc['collection']} | 维度: {vc['dim']}")
        logger.info(f"{'=' * 60}")

        # 清空目标集合
        from src.store.milvus_store import MilvusStore
        milvus = MilvusStore(collection_name=vc["collection"])
        milvus.connect()
        if milvus.count() > 0:
            logger.info(f"   清空已有数据 ({milvus.count()} 条)...")
            milvus.delete_all()
        milvus.close()

        # 加载 Embedder
        from src.embedder.embed import Embedder
        embedder = Embedder(model_path=vc["model_path"])
        logger.info(f"   Embedder 已加载: {vc['model_path']}")

        # 逐文档入库
        all_chunks = []
        for i, cl_path in enumerate(content_lists, 1):
            logger.info(f"\n   [{i}/{len(content_lists)}] {Path(cl_path).name}")
            chunks = process_one_document(
                content_list_path=cl_path,
                preview_only=False,
                no_bm25=True,  # BM25 最后统一构建
                embedder=embedder,
            )
            all_chunks.extend(chunks)

        logger.info(f"\n   ✅ {variant} 完成: {len(all_chunks)} 块")

    # 最后用 m3 的 chunks 构建 BM25（BM25 和 embedding 无关，复用 m3 的）
    logger.info(f"\n{'=' * 60}")
    logger.info(f"🔑 构建 BM25 索引...")
    _cfg.EMBED_VARIANT = "m3"
    _cfg.EMBEDDING_DIM = 1024
    _cfg.MILVUS_COLLECTION = "ccf_chunks"
    all_chunks_m3 = []
    embedder_m3 = None
    for cl_path in content_lists:
        chunks = process_one_document(
            content_list_path=cl_path,
            preview_only=True,
            embedder=None,
        )
        all_chunks_m3.extend(chunks)

    from src.store.keyword_store import BM25Index
    bm25 = BM25Index()
    bm25.build_index(all_chunks_m3)
    bm25_path = str(_cfg.DATA_DIR / "bm25_index.pkl")
    _cfg.DATA_DIR.mkdir(parents=True, exist_ok=True)
    bm25.save(bm25_path)
    logger.info(f"   BM25 索引已保存: {bm25_path} ({len(all_chunks_m3)} 条)")

    logger.info(f"\n{'#' * 60}")
    logger.info(f"✅ 全部完成！3 个集合：")
    logger.info(f"   ccf_chunks       (m3, 1024维)")
    logger.info(f"   ccf_chunks_base  (原始 bge-base, 768维)")
    logger.info(f"   ccf_chunks_base_ft (微调 bge-base, 768维)")
    logger.info(f"{'#' * 60}")


def main():
    parser = argparse.ArgumentParser(description="基于 MinerU 解析结果的离线入库（支持多文档）")
    parser.add_argument("--content-list", type=str, nargs="+", default=None,
                        help="一个或多个 content_list.json 路径（不传则自动发现所有文档）")
    parser.add_argument("--rebuild", action="store_true",
                        help="重建库（清空 Milvus 后重新入库）")
    parser.add_argument("--no-bm25", action="store_true",
                        help="跳过 BM25 索引构建")
    parser.add_argument("--preview", action="store_true",
                        help="只解析+分块，不入库，保存预览文件")
    parser.add_argument("--variant", type=str, default=None, choices=["base", "base_ft", "m3"],
                        help="Embedding 变体: base(原始bge-base) / base_ft(微调bge-base) / m3(bge-m3默认)")
    parser.add_argument("--all-variants", action="store_true",
                        help="一次性入库所有变体 (m3 + base + base_ft)，各自独立集合")
    args = parser.parse_args()

    # 应用变体配置
    if args.variant:
        from src.config import config as _cfg
        vc = _cfg.get_variant_config(args.variant)
        _cfg.EMBED_VARIANT = args.variant
        _cfg.EMBEDDING_DIM = vc["dim"]
        _cfg.MILVUS_COLLECTION = vc["collection"]
        logger.info(f"🔀 变体模式: {args.variant} | 模型: {vc['model_path']} | 集合: {vc['collection']} | 维度: {vc['dim']}")

    # --all-variants 模式：依次跑所有变体
    if args.all_variants:
        _run_all_variants(args)
        return

    # 1. 确定要处理的文件列表
    content_lists = args.content_list
    if not content_lists:
        content_lists = find_all_content_lists()

    if not content_lists:
        logger.error("❌ 未找到任何 MinerU content_list.json")
        logger.error("   请用 --content-list 指定路径，或将文件放到 data/source_docs/ 目录下")
        sys.exit(1)

    # 校验文件存在
    for p in content_lists:
        if not Path(p).exists():
            logger.error(f"❌ 文件不存在: {p}")
            sys.exit(1)

    logger.info(f"📚 待处理文档: {len(content_lists)} 个")
    for i, p in enumerate(content_lists, 1):
        logger.info(f"   [{i}] {Path(p).name}")

    # 2. 重建（清空一次）
    if args.rebuild:
        from src.store.milvus_store import MilvusStore
        from src.config import config as _cfg
        _coll = _cfg.MILVUS_COLLECTION
        logger.info(f"🗑️  清空 Milvus 集合: {_coll}...")
        milvus = MilvusStore(collection_name=_coll)
        milvus.connect()
        milvus.delete_all()
        milvus.close()
        logger.info("   已清空")

    # 3. 逐个处理文档
    all_chunks = []
    embedder = None  # 复用，避免每个文档都重新加载模型

    # 预加载 embedder（避免每个文档重新加载模型导致 C 级崩溃）
    if not args.preview:
        from src.embedder.embed import Embedder
        from src.config import config as _cfg
        _variant_cfg = _cfg.get_variant_config() if args.variant else None
        embedder = Embedder(model_path=_variant_cfg["model_path"] if _variant_cfg else None)
        if args.variant:
            logger.info(f"   Embedder 模型: {_variant_cfg['model_path']}")

    for i, content_list_path in enumerate(content_lists, 1):
        logger.info("")
        logger.info(f"{'#' * 60}")
        logger.info(f"# 文档 {i}/{len(content_lists)}: {Path(content_list_path).stem}")
        logger.info(f"{'#' * 60}")

        chunks = process_one_document(
            content_list_path=content_list_path,
            preview_only=args.preview,
            no_bm25=args.no_bm25,
            embedder=embedder,
        )

        all_chunks.extend(chunks)

    # 4. BM25 索引（合并所有文档的 chunks）
    if not args.preview and not args.no_bm25 and all_chunks:
        from src.store.keyword_store import BM25Index
        from src.config import config
        logger.info("")
        logger.info(f"🔑 构建 BM25 索引（合并 {len(content_lists)} 个文档，共 {len(all_chunks)} 块）...")
        bm25 = BM25Index()
        bm25.build_index(all_chunks)
        bm25_path = str(config.DATA_DIR / "bm25_index.pkl")
        config.DATA_DIR.mkdir(parents=True, exist_ok=True)
        bm25.save(bm25_path)
        logger.info(f"   BM25 索引已保存: {bm25_path}")

    # 5. 完成总结
    logger.info("")
    logger.info(f"{'=' * 60}")
    if args.preview:
        logger.info(f"✅ 预览模式完成，未入库。去掉 --preview 参数即可正常入库。")
    else:
        logger.info(f"✅ 全部入库完成！共处理 {len(content_lists)} 个文档，{len(all_chunks)} 个分块")
    logger.info(f"{'=' * 60}")


if __name__ == "__main__":
    main()
