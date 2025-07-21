from .base import BaseEmbedding
from .hf_embedding import CustomHFEmbedding
from .openai_interface import OpenAIEmbeddingsClient
from .zhipu_interface import ZhipuEmbeddings


class EmbeddingFactory:
    """
    参数化创建embedding client的工厂类
    """

    @staticmethod
    def create(emb_type: str, **params) -> BaseEmbedding:
        if emb_type == "openai":
            return OpenAIEmbeddingsClient(**params)
        elif emb_type == "zhipu":
            return ZhipuEmbeddings(**params)
        elif emb_type == "hf":
            return CustomHFEmbedding(**params)
        else:
            raise ValueError(f"不支持的embedding类型: {emb_type}")
