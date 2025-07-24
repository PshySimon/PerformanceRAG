from ...pipeline.registry import ComponentRegistry
from .es_retriever import ESRetrieverComponent
from .bm25_retriever import BM25RetrieverComponent
from .base_retriever import BaseRetrieverComponent
from .retriever_factory import RetrieverFactory

# 注册组件
ComponentRegistry.register("retriever", "elasticsearch")(ESRetrieverComponent)
ComponentRegistry.register("retriever", "es")(ESRetrieverComponent)
ComponentRegistry.register("retriever", "bm25")(BM25RetrieverComponent)

__all__ = [
    "ESRetrieverComponent",
    "BM25RetrieverComponent", 
    "BaseRetrieverComponent",
    "RetrieverFactory"
]