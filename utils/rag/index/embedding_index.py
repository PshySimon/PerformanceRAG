import logging

from llama_index.core import StorageContext, VectorStoreIndex, load_index_from_storage
from tqdm import tqdm

from .base_index import BaseIndex

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


class EmbeddingIndex(BaseIndex):
    def __init__(self, embedding_client, index=None):
        self.embedding_client = embedding_client
        self.index = index

    def build(self, documents, batch_size=16, save_path=None) -> bool:
        try:
            # 关闭transformers/llama_index内部进度条
            import os
            os.environ["TRANSFORMERS_NO_PROGRESS_BAR"] = "1"
            # 批量embedding
            texts = [doc.text for doc in documents]
            embeddings = []
            pbar = tqdm(total=len(texts), desc="Embedding Documents (outer)")
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                batch_embs = self.embedding_client.embed_texts(batch_texts)
                embeddings.extend(batch_embs)
                pbar.update(len(batch_texts))
            pbar.close()
            try:
                self.index = VectorStoreIndex.from_documents(
                    documents, embed_model=self.embedding_client
                )
                # 自动保存索引
                if save_path:
                    self.save(save_path)
                return True
            except Exception as e:
                print(f"[ERROR] index build failed: {e}")
                return False
        except Exception as e:
            print(f"[ERROR] embedding failed: {e}")
            return False

    def save(self, path) -> bool:
        try:
            if isinstance(self.index, VectorStoreIndex):
                self.index.storage_context.persist(path)
                return True
            else:
                return False
        except Exception:
            return False

    @classmethod
    def load(cls, path, embedding_client) -> "EmbeddingIndex|None":
        try:
            storage_context = StorageContext.from_defaults(persist_dir=path)
            index = load_index_from_storage(
                storage_context, embed_model=embedding_client
            )
            return cls(embedding_client, index)
        except Exception:
            return None

    def get_search_index(self):
        return self.index

    @classmethod
    def is_serialized(cls, path: str, embedding_client) -> bool:
        try:
            return cls.load(path, embedding_client) is not None
        except Exception:
            return False
