from utils.config import config
from .naive_rag_pipeline import NaiveRagPipeline
from .advanced_rag_pipeline import AdvancedRagPipeline

# 全局缓存，避免重复prepare
_pipeline_cache = {}

def create_pipeline(pipeline_type: str):
    if pipeline_type in _pipeline_cache:
        return _pipeline_cache[pipeline_type]
    pipeline_cfg = config.pipeline[pipeline_type]
    if pipeline_type == "naive_rag":
        pipeline = NaiveRagPipeline(pipeline_cfg)
    elif pipeline_type == "advanced_rag":
        pipeline = AdvancedRagPipeline(pipeline_cfg)
    else:
        raise ValueError(f"不支持的pipeline类型: {pipeline_type}")
    pipeline.prepare()
    _pipeline_cache[pipeline_type] = pipeline
    return pipeline 