from .embedding_reranker import EmbeddingReranker
from utils.config import config
from typing import Optional, Dict, Any

def create_reranker(reranker_type: Optional[str] = None, params: Optional[Dict[str, Any]] = None):
    if reranker_type is None:
        reranker_type = config.reranker.clients[config.reranker.default]["type"]
        cfg = config.reranker.clients[config.reranker.default]
    else:
        cfg = None
        for k, v in config.reranker.clients.items():
            if v["type"] == reranker_type:
                cfg = v
                break
        if cfg is None:
            cfg = {}
    params = params or {}
    if cfg:
        config_params = {k: v for k, v in cfg.items() if k != "type"}
        config_params.update(params)
    else:
        config_params = params
    if reranker_type == "embedding":
        return EmbeddingReranker(**config_params)
    raise ValueError(f"不支持的Reranker类型: {reranker_type}") 