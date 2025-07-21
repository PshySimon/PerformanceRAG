from zhipuai import ZhipuAI
from .base import BaseEmbedding


class ZhipuEmbeddings(BaseEmbedding):
    def __init__(self, api_key: str, model: str, timeout: float = 5.0):
        self._api_key = api_key
        self._model = model
        self._timeout = timeout
        self._client = ZhipuAI(api_key=self._api_key, timeout=self._timeout)

    def embed_text(self, text: str) -> list[float]:
        resp = self._client.embeddings.create(model=self._model, input=text)
        return resp.data[0].embedding

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        resp = self._client.embeddings.create(model=self._model, input=texts)
        return [item.embedding for item in resp.data]

    def _get_text_embedding(self, text: str):
        return self.embed_text(text)

    def _get_query_embedding(self, query: str):
        return self.embed_text(query)

    async def _aget_query_embedding(self, query: str):
        return self.embed_text(query)
