"""
API 数据模型
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional


class QueryRequest(BaseModel):
    question: str = Field(..., description="用户问题")
    session_id: str = Field(default="default", description="会话 ID，用于多轮对话")
    engine: str = Field(default="auto", description="检索引擎: auto(智能选择) / traditional(传统RAG) / lightrag(图谱RAG)")
    lightrag_mode: str = Field(default="mix", description="LightRAG查询模式: local/global/hybrid/mix/naive")


class Source(BaseModel):
    page_no: int
    text: str
    score: float
    chunk_type: str = "text"
    section_path: str = ""
    source_file: str = ""


class QueryResponse(BaseModel):
    answer: str
    sources: List[Source]
    session_id: str


class HealthResponse(BaseModel):
    status: str = "ok"
    milvus: str = "unknown"
    redis: str = "unknown"


class UploadResponse(BaseModel):
    status: str
    message: str
    chunks_count: int = 0
    file_name: str = ""
