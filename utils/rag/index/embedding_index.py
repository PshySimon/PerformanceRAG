import logging
import os
import numpy as np
import faiss
import joblib
from tqdm import tqdm
from .base_index import BaseIndex

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

class EmbeddingIndex(BaseIndex):
    def __init__(self, embedding_client, index=None, metadata=None):
        self.embedding_client = embedding_client
        self.index = index  # faiss.Index
        self.metadata = metadata or []  # list of dict or Document

    def build(self, documents, batch_size=16, save_path=None) -> bool:
        try:
            texts = [doc.text for doc in documents]
            embeddings = []
            pbar = tqdm(total=len(texts), desc="Embedding Documents (outer)")
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                batch_embs = self.embedding_client.embed_texts(batch_texts)
                embeddings.extend(batch_embs)
                pbar.update(len(batch_texts))
            pbar.close()
            embeddings = np.array(embeddings).astype('float32')
            dim = embeddings.shape[1]
            self.index = faiss.IndexFlatL2(dim)
            self.index.add(embeddings)
            # 保存 metadata（可直接存 Document 或 dict）
            self.metadata = [getattr(doc, 'metadata', {}) for doc in documents]
            if save_path:
                self.save(save_path)
            return True
        except Exception as e:
            print(f"[ERROR] embedding/build failed: {e}")
            return False

    def save(self, path) -> bool:
        try:
            os.makedirs(path, exist_ok=True)
            faiss.write_index(self.index, os.path.join(path, "faiss.index"))
            joblib.dump(self.metadata, os.path.join(path, "metadata.joblib"))
            return True
        except Exception as e:
            print(f"[ERROR] save failed: {e}")
            return False

    @classmethod
    def load(cls, path, embedding_client) -> "EmbeddingIndex|None":
        try:
            index = faiss.read_index(os.path.join(path, "faiss.index"))
            metadata = joblib.load(os.path.join(path, "metadata.joblib"))
            return cls(embedding_client, index, metadata)
        except Exception as e:
            print(f"[ERROR] load failed: {e}")
            return None

    def get_search_index(self):
        return self.index

    @classmethod
    def is_serialized(cls, path: str, embedding_client=None) -> bool:
        try:
            faiss_path = os.path.join(path, "faiss.index")
            meta_path = os.path.join(path, "metadata.joblib")
            return os.path.exists(faiss_path) and os.path.exists(meta_path)
        except Exception:
            return False
