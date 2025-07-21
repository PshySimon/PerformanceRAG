from typing import Any, Dict, Optional

from utils.config import config

from .bm25_retriever import BM25Retriever
from .embedding_retriever import EmbeddingRetriever
from .hybrid_retriever import HybridRetriever


def create_retriever(
    retriever_type: Optional[str] = None, params: Optional[Dict[str, Any]] = None
):
    # 优先参数，其次配置
    if retriever_type is None:
        retriever_type = config.retriever.clients[config.retriever.default]["type"]
        cfg = config.retriever.clients[config.retriever.default]
    else:
        # 支持直接传type
        cfg = None
        for k, v in config.retriever.clients.items():
            if v["type"] == retriever_type:
                cfg = v
                break
        if cfg is None:
            cfg = {}
    params = params or {}
    # 合并配置参数（除type）
    if cfg:
        config_params = {k: v for k, v in cfg.items() if k != "type"}
        config_params.update(params)
    else:
        config_params = params
    if retriever_type == "embedding":
        return EmbeddingRetriever(**config_params)
    if retriever_type == "bm25":
        return BM25Retriever(**config_params)
    if retriever_type == "hybrid":
        hybrid_params = {k: v for k, v in config_params.items() if k != "type"}
        return HybridRetriever(**hybrid_params)
    raise ValueError(f"不支持的Retriever类型: {retriever_type}")
