import concurrent.futures
import os
import pickle
from collections import defaultdict
from typing import List
import re
import jieba

from llama_index.core.schema import Document
from tqdm import tqdm

from utils.config import config

from .base_index import BaseIndex


class BM25Index(BaseIndex):
    def __init__(self, index_data=None, doc_store=None):
        self.index_data = index_data or defaultdict(list)  # term -> list of doc_ids
        self.doc_store = doc_store or {}  # doc_id -> Document
        self.doc_lengths = {}  # doc_id -> length
        self.avg_doc_length = 0
        self.N = 0  # total docs

    def _tokenize(self, text: str) -> List[str]:
        # 先用jieba分词，再用正则分割英文和符号
        tokens = list(jieba.cut(text, cut_all=False))
        final_tokens = []
        for token in tokens:
            # 对每个jieba分出来的词再用正则分割
            final_tokens.extend([t for t in re.split(r'\W+', token) if t])
        return final_tokens

    def build(self, documents: List[Document]) -> bool:
        try:
            path = config.index.clients.bm25["persist_dir"]
            if self.is_serialized(path):
                loaded = self.load(path)
                if loaded is None:
                    return False
                self.index_data = loaded.index_data
                self.doc_store = loaded.doc_store
                self.doc_lengths = loaded.doc_lengths
                self.avg_doc_length = loaded.avg_doc_length
                self.N = loaded.N
                return True

            index_cfg = config.index.clients[config.index.default]
            max_worker = index_cfg.get("max_worker", 8)

            self.index_data = defaultdict(list)
            self.doc_store = {}
            self.doc_lengths = {}
            self.N = len(documents)
            total_length = 0

            def process_doc(i_doc):
                i, doc = i_doc
                doc_id = str(i)
                words = self._tokenize(doc.text)
                return doc_id, doc, words, len(words)

            with concurrent.futures.ThreadPoolExecutor(
                max_workers=max_worker
            ) as executor:
                results = list(
                    tqdm(
                        executor.map(process_doc, enumerate(documents)),
                        total=len(documents),
                        desc="BM25 Indexing",
                    )
                )
            for doc_id, doc, words, length in results:
                self.doc_store[doc_id] = doc
                self.doc_lengths[doc_id] = length
                total_length += length
                for word in set(words):
                    self.index_data[word].append(doc_id)
            self.avg_doc_length = total_length / self.N if self.N > 0 else 0
            return True
        except Exception as e:
            print(f"[ERROR] bm25 build failed: {e}")
            return False

    def save(self, path) -> bool:
        try:
            if self.is_serialized(path):
                return True
            os.makedirs(path, exist_ok=True)
            with open(os.path.join(path, "bm25_index.pkl"), "wb") as f:
                pickle.dump(
                    {
                        "index_data": dict(self.index_data),
                        "doc_store": self.doc_store,
                        "doc_lengths": self.doc_lengths,
                        "avg_doc_length": self.avg_doc_length,
                        "N": self.N,
                    },
                    f,
                )
            return True
        except Exception:
            return False

    @classmethod
    def load(cls, path) -> "BM25Index|None":
        try:
            with open(os.path.join(path, "bm25_index.pkl"), "rb") as f:
                data = pickle.load(f)
            obj = cls()
            obj.index_data = defaultdict(list, data["index_data"])
            obj.doc_store = data["doc_store"]
            obj.doc_lengths = data["doc_lengths"]
            obj.avg_doc_length = data["avg_doc_length"]
            obj.N = data["N"]
            return obj
        except Exception:
            return None

    @classmethod
    def is_serialized(cls, path: str) -> bool:
        try:
            return cls.load(path) is not None
        except Exception:
            return False

    def get_search_index(self):
        return self
