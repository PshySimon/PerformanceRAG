from typing import Optional

from llama_index.core import VectorStoreIndex

from .base_retriever import BaseRetriever


class EmbeddingRetriever(BaseRetriever):
    index_type = "embedding"
    def __init__(self, similarity_metric: str = "cosine", top_k: int = 5):
        self.similarity_metric = similarity_metric
        self.top_k = top_k
        # kwargs预留扩展

    def retrieve(self, index, query_str: str, top_k: Optional[int] = None):
        # 自动兼容index类型
        if hasattr(index, "get_search_index"):
            search_index = index.get_search_index()
        else:
            search_index = index
        if not isinstance(search_index, VectorStoreIndex):
            raise RuntimeError("当前index对象不支持检索操作")
        use_top_k = top_k if top_k is not None else self.top_k
        retriever = search_index.as_retriever(similarity_top_k=use_top_k)
        # similarity_metric参数可用于后续扩展
        return retriever.retrieve(query_str)
