from .base_router import BaseRouter
from typing import List, Any
from transformers import BertTokenizer, BertModel
import torch
import numpy as np

class BertRouter(BaseRouter):
    def __init__(self, model_name: str = "bert-base-uncased"):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
        self.model.eval()

    def get_embedding(self, text: str):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        with torch.no_grad():
            outputs = self.model(**inputs)
            emb = outputs.last_hidden_state[:, 0, :].squeeze(0).cpu().numpy()
        return emb

    def route(self, query: str, candidates: List[Any], **kwargs) -> Any:
        query_emb = self.get_embedding(query)
        best_idx = 0
        best_score = -float('inf')
        for i, c in enumerate(candidates):
            text = getattr(c, 'text', getattr(c, 'get_content', lambda: "")())
            cand_emb = self.get_embedding(text)
            score = np.dot(query_emb, cand_emb) / (np.linalg.norm(query_emb) * np.linalg.norm(cand_emb) + 1e-8)
            if score > best_score:
                best_score = score
                best_idx = i
        return candidates[best_idx] 