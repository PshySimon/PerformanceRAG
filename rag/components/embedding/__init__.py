from ...pipeline.registry import ComponentRegistry
from .openai_embedding import OpenAIEmbeddingComponent
from .base_embedding import BaseEmbedding
from .openai_embedding import OpenAIEmbedding
from .hf_embedding import HFEmbedding
from .embedding_factory import EmbeddingFactory

# 注册组件
ComponentRegistry.register("embedding", "openai")(OpenAIEmbeddingComponent)

__all__ = [
    "BaseEmbedding",
    "OpenAIEmbedding",
    "HFEmbedding",
    "OpenAIEmbeddingComponent",
    "EmbeddingFactory",
]