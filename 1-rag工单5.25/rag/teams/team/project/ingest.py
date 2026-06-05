"""
PDF 离线入库脚本 — 一次运行，入库完成即结束
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统

用法：
  python ingest.py                                          # 自动检测默认 PDF 并入库
  python ingest.py --pdf a.pdf b.pdf                        # 指定多个 PDF
  python ingest.py --pdf-dir "D:\Desktop\附件"              # 入库目录下所有 PDF
  python ingest.py --rebuild                                # 清空 Milvus 后重新入库
  python ingest.py --no-bm25                                # 跳过 BM25 索引
"""

import sys
import argparse
from pathlib import Path

# 上线 TensorFlow 噪音
import os
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

# 路径修复
_PROJECT_ROOT = Path(__file__).parent.resolve()
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.logger import logger


def ingest_one(pdf_path: str, embedder=None, milvus=None):
    """入库单个 PDF，返回 (chunks_count, inserted_count)"""
    from src.loader.pdf_loader import PDFLoader
    from src.chunker.text_splitter import Chunker

    logger.info(f"Processing: {Path(pdf_path).name}")
    loader = PDFLoader(pdf_path)
    pages = loader.extract_pages()
    logger.info(f"   pages: {len(pages)}")

    chunker = Chunker()
    chunks = chunker.chunk_pages(pages, source_file=Path(pdf_path).name)
    logger.info(f"   chunks: {len(chunks)}")

    if not chunks:
        logger.warning(f"   skip (no chunks): {Path(pdf_path).name}")
        return 0, 0

    # 向量化（复用 embedder 实例）
    if embedder is None:
        from src.embedder.embed import Embedder
        embedder = Embedder()
    texts = [c["text"] for c in chunks]
    embeddings = embedder.embed_batch(texts)
    logger.info(f"   embedded: {len(embeddings)} (dim={len(embeddings[0])})")

    # 入库（复用 milvus 连接）
    if milvus is None:
        from src.store.milvus_store import MilvusStore
        milvus = MilvusStore()
        milvus.connect()
    inserted = milvus.insert_chunks(chunks, embeddings)

    return len(chunks), inserted


def get_default_pdfs():
    """从 config 的候选路径获取所有默认 PDF"""
    from src.config import config
    candidates = [
        config.PROJECT_ROOT / "招股说明书1-无水印.pdf",
        config.PROJECT_ROOT.parent.parent / "招股说明书1-无水印.pdf",
        config.PROJECT_ROOT.parent.parent.parent / "招股说明书1-无水印.pdf",
        Path(r"D:\Desktop\专高六工单\RAG 工单\RAG 工单\附件\招股说明书1-无水印.pdf"),
        Path(r"D:\Desktop\专高六工单\RAG 工单\RAG 工单\附件\招股说明书2.pdf"),
    ]
    env_path = os.environ.get("DEFAULT_PDF_PATH", "")
    if env_path:
        candidates.insert(0, Path(env_path).resolve())
    return [str(p) for p in candidates if p.exists()]


def main():
    parser = argparse.ArgumentParser(description="PDF 离线入库")
    parser.add_argument("--pdf", type=str, nargs="+", default=None,
                        help="PDF 文件路径（支持多个，空格分隔）")
    parser.add_argument("--pdf-dir", type=str, default=None,
                        help="PDF 目录（入库目录下所有 .pdf 文件）")
    parser.add_argument("--rebuild", action="store_true",
                        help="重建库（清空 Milvus 后重新入库）")
    parser.add_argument("--no-bm25", action="store_true",
                        help="跳过 BM25 索引构建")
    args = parser.parse_args()

    # 1. 确定 PDF 路径列表
    pdf_paths = []

    if args.pdf:
        pdf_paths = args.pdf
    elif args.pdf_dir:
        pdf_dir = Path(args.pdf_dir)
        if not pdf_dir.is_dir():
            logger.error(f"not a dir: {args.pdf_dir}")
            sys.exit(1)
        pdf_paths = [str(p) for p in sorted(pdf_dir.glob("*.pdf"))]
    else:
        pdf_paths = get_default_pdfs()

    if not pdf_paths:
        logger.error("no PDF found, use --pdf or --pdf-dir")
        sys.exit(1)

    # 校验文件存在
    for p in pdf_paths:
        if not Path(p).exists():
            logger.error(f"file not found: {p}")
            sys.exit(1)

    logger.info(f"total {len(pdf_paths)} PDF(s) to ingest")

    # 2. 重建
    if args.rebuild:
        from src.config import config
        from src.store.milvus_store import MilvusStore
        logger.info("clearing Milvus...")
        milvus = MilvusStore()
        milvus.connect()
        milvus.delete_all()
        milvus.close()
        logger.info("   cleared")

    # 3. 批量入库
    from src.embedder.embed import Embedder
    from src.store.milvus_store import MilvusStore

    embedder = Embedder()
    milvus = MilvusStore()
    milvus.connect()

    total_chunks = 0
    total_inserted = 0

    for i, pdf_path in enumerate(pdf_paths, 1):
        logger.info(f"[{i}/{len(pdf_paths)}] {Path(pdf_path).name}")
        chunks_count, inserted_count = ingest_one(pdf_path, embedder=embedder, milvus=milvus)
        total_chunks += chunks_count
        total_inserted += inserted_count

    final_total = milvus.count()
    milvus.close()

    logger.info("=" * 50)
    logger.info(f"done! {len(pdf_paths)} PDF(s), {total_chunks} chunks, {total_inserted} inserted (total: {final_total})")


if __name__ == "__main__":
    main()
