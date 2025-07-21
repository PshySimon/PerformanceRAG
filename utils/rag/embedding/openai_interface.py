# utils/rag/embedding/openai_client.py

import os
import openai
from utils.config import config  # 假设你已实现 config.py
from .base import BaseEmbedding


class OpenAIEmbeddingsClient(BaseEmbedding):
    """OpenAI embedding客户端，支持单条和批量文本embedding"""
    def __init__(self):
        """初始化OpenAI embedding客户端"""
        emb_cfg = config.embeddings.clients.openai
        api_key = emb_cfg.api_key
        if api_key and api_key.startswith("${") and api_key.endswith("}"):
            api_key = os.getenv(api_key[2:-1])
        openai.api_key = api_key
        self.model = emb_cfg.model
        self.batch_size = emb_cfg.batch_size

        if not openai.api_key:
            raise ValueError("OPENAI_API_KEY is not set or invalid")

    def embed_text(self, text: str) -> list[float]:
        """对单条文本进行embedding"""
        resp = openai.Embedding.create(model=self.model, input=text)  # pylint: disable=no-member
        return resp["data"][0]["embedding"]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """对多条文本进行批量embedding"""
        results = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            resp = openai.Embedding.create(model=self.model, input=batch)  # pylint: disable=no-member
            results.extend([d["embedding"] for d in resp["data"]])
        return results

    def _get_text_embedding(self, text: str):
        return self.embed_text(text)

    def _get_query_embedding(self, query: str):
        return self.embed_text(query)

    async def _aget_query_embedding(self, query: str):
        return self.embed_text(query)
