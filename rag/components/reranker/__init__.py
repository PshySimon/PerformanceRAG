from ...pipeline.registry import ComponentRegistry
from .base_reranker import BaseRerankerComponent
from .embedding_reranker import EmbeddingRerankerComponent
from .llm_reranker import LLMRerankerComponent
from .zhipu_reranker import ZhipuRerankerComponent
from .openai_reranker import OpenAIRerankerComponent
from .reranker_factory import RerankerFactory

# 注册组件
ComponentRegistry.register("reranker", "llm")(LLMRerankerComponent)
ComponentRegistry.register("reranker", "embedding")(EmbeddingRerankerComponent)
ComponentRegistry.register("reranker", "zhipu")(ZhipuRerankerComponent)
ComponentRegistry.register("reranker", "openai")(OpenAIRerankerComponent)

__all__ = [
    "BaseRerankerComponent",
    "LLMRerankerComponent",
    "EmbeddingRerankerComponent",
    "ZhipuRerankerComponent",
    "OpenAIRerankerComponent",
    "RerankerFactory",
]
