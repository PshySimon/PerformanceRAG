from .rewrite import RewriteExpansion
from .hyde import HyDEExpansion
from .multi_query import MultiQueryExpansion
from .decompose import DecomposeExpansion
from .disambiguate import DisambiguateExpansion
from .abstract import AbstractExpansion
from utils.config import config
from typing import Dict, Any, Optional

def create_query_expansion(expansion_type: str, params: Optional[Dict[str, Any]] = None):
    if not expansion_type:
        raise ValueError("必须显式指定expansion_type")
    cfg = None
    for k, v in config.query.modules.items():
        if v["type"] == expansion_type:
            cfg = v
            break
    if cfg is None:
        cfg = {}
    params = params or {}
    config_params = {k: v for k, v in cfg.items() if k != "type"}
    config_params.update(params)
    config_params.pop("type", None)
    if expansion_type == "rewrite":
        return RewriteExpansion(**config_params)
    if expansion_type == "hyde":
        return HyDEExpansion(**config_params)
    if expansion_type == "multi_query":
        return MultiQueryExpansion(**config_params)
    if expansion_type == "decompose":
        return DecomposeExpansion(**config_params)
    if expansion_type == "disambiguate":
        return DisambiguateExpansion(**config_params)
    if expansion_type == "abstract":
        return AbstractExpansion(**config_params)
    raise ValueError(f"不支持的QueryExpansion类型: {expansion_type}") 