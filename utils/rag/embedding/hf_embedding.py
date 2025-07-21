from llama_index.core.base.embeddings.base import BaseEmbedding
from transformers import AutoTokenizer, AutoModel
import torch
from pydantic import PrivateAttr

class CustomHFEmbedding(BaseEmbedding):
    _tokenizer: any = PrivateAttr()
    _model: any = PrivateAttr()
    _device: str = PrivateAttr()

    def __init__(self, model_name: str, device: str = None):
        super().__init__()
        self.model_name = model_name
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModel.from_pretrained(model_name)
        if device:
            self._device = device
        elif torch.cuda.is_available():
            self._device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self._device = "mps"
        else:
            self._device = "cpu"
        self._model.to(self._device)
        self._model.eval()

    @torch.no_grad()
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        inputs = self._tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True, max_length=512
        ).to(self._device)
        outputs = self._model(**inputs)
        embs = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        return [emb.tolist() for emb in embs]

    @torch.no_grad()
    def embed_query(self, text: str) -> list[float]:
        inputs = self._tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512
        ).to(self._device)
        outputs = self._model(**inputs)
        emb = outputs.last_hidden_state[:, 0, :].squeeze(0).cpu().numpy()
        return emb.tolist()

    @torch.no_grad()
    def embed_text(self, text: str) -> list[float]:
        # 兼容旧接口
        return self.embed_query(text)

    @torch.no_grad()
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        # 兼容旧接口
        return self.embed_documents(texts)

    def _get_text_embedding(self, text: str):
        return self.embed_query(text)

    def _get_query_embedding(self, query: str):
        return self.embed_query(query)

    async def _aget_query_embedding(self, query: str):
        return self.embed_query(query)

    @classmethod
    def class_name(cls) -> str:
        return "custom_hf"
