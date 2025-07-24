# rag/pipeline/factory.py
from utils.logger import get_logger

from .builder import PipelineBuilder
from .executor import PipelineExecutor

# 全局缓存，避免重复构建
_pipeline_cache = {}


def create_pipeline(pipeline_name: str, use_cache: bool = True) -> PipelineExecutor:
    """创建并返回Pipeline执行器"""
    logger = get_logger(__name__)

    # 检查缓存
    if use_cache and pipeline_name in _pipeline_cache:
        logger.debug(f"从缓存获取Pipeline: {pipeline_name}")
        return _pipeline_cache[pipeline_name]

    try:
        # 使用Builder构建Pipeline
        builder = PipelineBuilder.from_config(pipeline_name)
        pipeline = builder.build()

        # 缓存Pipeline
        if use_cache:
            _pipeline_cache[pipeline_name] = pipeline

        logger.info(f"成功创建Pipeline: {pipeline_name}")
        return pipeline

    except Exception as e:
        logger.error(f"创建Pipeline失败: {pipeline_name}, 错误: {e}")
        raise


def clear_cache():
    """清空Pipeline缓存"""
    global _pipeline_cache
    _pipeline_cache.clear()


def get_cached_pipelines():
    """获取已缓存的Pipeline列表"""
    return list(_pipeline_cache.keys())


def build_pipeline(pipeline_name: str) -> PipelineBuilder:
    """创建Pipeline Builder实例（不执行build）"""
    return PipelineBuilder.from_config(pipeline_name)
