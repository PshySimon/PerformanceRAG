from typing import List
from llama_index.core.base.embeddings.base import BaseEmbedding as LlamaBaseEmbedding


class BaseEmbedding(LlamaBaseEmbedding):
    def embed_text(self, text: str) -> List[float]:
        """对单条文本进行embedding"""
        raise NotImplementedError

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """对多条文本进行批量embedding"""
        raise NotImplementedError
