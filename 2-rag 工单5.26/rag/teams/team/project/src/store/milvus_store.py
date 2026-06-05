"""
Milvus 向量库操作
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
"""

from typing import List, Dict, Any
from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)
from src.config import config


class MilvusStore:
    """Milvus 向量库 CRUD"""

    def __init__(
        self,
        host: str | None = None,
        port: int | None = None,
        collection_name: str | None = None,
    ):
        self.host = host or config.MILVUS_HOST
        self.port = port or config.MILVUS_PORT
        self.collection_name = collection_name or config.MILVUS_COLLECTION
        self._connected = False
        self._collection: Collection | None = None

    def connect(self) -> None:
        """连接 Milvus 并确保 Collection 存在"""
        if self._connected:
            return
        connections.connect(
            alias="default",
            host=self.host,
            port=self.port,
        )
        self._ensure_collection()
        self._connected = True

    def _ensure_collection(self) -> None:
        """确保 Collection 存在且 schema 匹配"""
        from src.config import config
        expected_fields = ["id", "vector", "text", "page_no", "source_file", "chunk_index"]
        expected_dim = config.EMBEDDING_DIM

        if utility.has_collection(self.collection_name):
            self._collection = Collection(self.collection_name)
            # 验证 schema 字段名和维度
            existing = [f.name for f in self._collection.schema.fields]
            existing_dims = {
                f.params.get("dim") for f in self._collection.schema.fields
                if f.dtype == DataType.FLOAT_VECTOR
            }
            dim_ok = (len(existing_dims) == 1 and expected_dim in existing_dims)

            if existing != expected_fields or not dim_ok:
                import logging
                log = logging.getLogger("rag")
                log.warning(
                    f"Milvus schema 不匹配，重建 Collection\n"
                    f"  现有字段: {existing} dim={existing_dims}\n"
                    f"  期望字段: {expected_fields} dim={expected_dim}"
                )
                self._collection.drop()
                return self._create_collection()
            return

        self._create_collection()

    def _create_collection(self) -> None:
        """创建 Collection + 索引"""
        from src.config import config
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=config.EMBEDDING_DIM),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=8192),
            FieldSchema(name="page_no", dtype=DataType.INT64),
            FieldSchema(name="source_file", dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name="chunk_index", dtype=DataType.INT64),
        ]
        schema = CollectionSchema(fields, description="PDF文档分块向量库")
        self._collection = Collection(name=self.collection_name, schema=schema)

        index_params = {
            "metric_type": "IP",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 128},
        }
        self._collection.create_index(field_name="vector", index_params=index_params)
        import logging
        logging.getLogger("rag").info(f"Milvus Collection '{self.collection_name}' 已创建")

    def insert_chunks(
        self,
        chunks: List[Dict[str, Any]],
        embeddings: List[List[float]],
    ) -> int:
        """批量插入分块

        Args:
            chunks: [{"text", "page_no", "source_file", "chunk_index"}, ...]
            embeddings: 对应的向量列表

        Returns:
            插入数量
        """
        self.connect()
        entities = [
            embeddings,
            [c["text"] for c in chunks],
            [c["page_no"] for c in chunks],
            [c["source_file"] for c in chunks],
            [c["chunk_index"] for c in chunks],
        ]
        insert_result = self._collection.insert(entities)
        self._collection.flush()
        return len(insert_result.primary_keys)

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 20,
    ) -> List[Dict[str, Any]]:
        """向量检索

        Args:
            query_embedding: 查询向量
            top_k: 返回数量

        Returns:
            [{"text": str, "page_no": int, "source_file": str, "score": float}, ...]
        """
        self.connect()
        self._collection.load()

        search_params = {
            "metric_type": "IP",
            "params": {"nprobe": 10},
        }
        results = self._collection.search(
            data=[query_embedding],
            anns_field="vector",
            param=search_params,
            limit=top_k,
            output_fields=["text", "page_no", "source_file", "chunk_index"],
        )

        hits = []
        for hit in results[0]:
            hits.append({
                "text": hit.entity.get("text"),
                "page_no": hit.entity.get("page_no"),
                "source_file": hit.entity.get("source_file"),
                "score": hit.score,
            })
        return hits

    def count(self) -> int:
        """文档数量"""
        self.connect()
        self._collection.flush()
        return self._collection.num_entities

    def delete_all(self) -> None:
        """清空 Collection"""
        self.connect()
        self._collection.drop()
        self._collection = None
        self._connected = False

    def close(self) -> None:
        """断开连接"""
        if self._connected:
            connections.disconnect("default")
            self._connected = False
