from utils.rag.embedding.base import BaseEmbedding
from .embedding_index import EmbeddingIndex
from .bm25_index import BM25Index


def create_index(index_type: str, embedding_client: BaseEmbedding = None):
    if index_type == "embedding":
        if embedding_client is None:
            raise ValueError("embedding_client must be provided for EmbeddingIndex")
        return EmbeddingIndex(embedding_client=embedding_client)
    if index_type == "bm25":
        return BM25Index()
    # 可扩展更多类型
    raise ValueError(f"不支持的Index类型: {index_type}")
