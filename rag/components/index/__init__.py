from ...pipeline.registry import ComponentRegistry
from .es_indexer import ESIndexerComponent
from .bm25_indexer import BM25IndexerComponent
from .base_indexer import BaseIndexer

# 注册组件
ComponentRegistry.register("indexer", "elasticsearch")(ESIndexerComponent)
ComponentRegistry.register("indexer", "bm25")(BM25IndexerComponent)

__all__ = [
    "ESIndexerComponent",
    "BM25IndexerComponent",
    "BaseIndexer"
]