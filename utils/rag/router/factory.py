from .llm_router import LLMRouter
from .bert_router import BertRouter
from utils.config import config
from typing import Optional, Dict, Any

def create_router(router_type: Optional[str] = None, params: Optional[Dict[str, Any]] = None):
    if router_type is None:
        router_type = config.router.clients[config.router.default]["type"]
        cfg = config.router.clients[config.router.default]
    else:
        cfg = None
        for k, v in config.router.clients.items():
            if v["type"] == router_type:
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
    if router_type == "llm":
        return LLMRouter(**config_params)
    if router_type == "embedding":
        return BertRouter(**config_params)
    raise ValueError(f"不支持的Router类型: {router_type}") 