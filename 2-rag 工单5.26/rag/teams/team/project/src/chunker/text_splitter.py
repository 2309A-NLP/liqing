"""
文档分块 — RecursiveCharacterTextSplitter
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
"""

from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter


class Chunker:
    """文档分块器，将页列表切分为重叠块"""

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 128,
    ):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", "。", ".", " ", ""],
            length_function=len,
        )

    def chunk_pages(
        self,
        pages: List[Dict[str, Any]],
        source_file: str = "",
    ) -> List[Dict[str, Any]]:
        """将页列表切分为块

        Args:
            pages: PDFLoader.extract_pages() 的输出
            source_file: 源文件名

        Returns:
            [{"text": str, "page_no": int, "source_file": str, "chunk_index": int}, ...]
        """
        chunks = []
        chunk_index = 0

        # 先按页把文本和页码配对
        for page in pages:
            text = page["text"]
            page_no = page["page_no"]

            if not text.strip():
                continue

            # 对每页单独分块
            split_docs = self.splitter.create_documents([text])
            for doc in split_docs:
                chunks.append({
                    "text": doc.page_content,
                    "page_no": page_no,
                    "source_file": source_file,
                    "chunk_index": chunk_index,
                })
                chunk_index += 1

        return chunks

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
            })
        return chunks
