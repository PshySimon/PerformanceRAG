from typing import Any, Dict

from utils.logger import get_logger

from .embedding_reranker import EmbeddingRerankerComponent
from .llm_reranker import LLMRerankerComponent


class RerankerFactory:
    """重排组件工厂"""

    @staticmethod
    def create_reranker(name: str, reranker_type: str, config: Dict[str, Any]):
        """创建重排组件"""
        logger = get_logger(__name__)

        if reranker_type == "llm":
            logger.info(f"创建LLM重排组件: {name}")
            return LLMRerankerComponent(name, config)
        elif reranker_type == "embedding":
            logger.info(f"创建Embedding重排组件: {name}")
            return EmbeddingRerankerComponent(name, config)
        else:
            raise ValueError(f"不支持的重排组件类型: {reranker_type}")

    @staticmethod
    def create_pipeline_rerankers(pipeline_config: Dict[str, Any]) -> Dict[str, Any]:
        """根据流水线配置创建多个重排组件"""
        rerankers = {}

        for component_config in pipeline_config.get("components", []):
            if component_config.get("type") == "reranker":
                name = component_config["name"]
                subtype = component_config["subtype"]
                config = component_config.get("config", {})

                reranker = RerankerFactory.create_reranker(name, subtype, config)
                rerankers[name] = reranker

        return rerankers
