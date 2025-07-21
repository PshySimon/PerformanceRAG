import math
import re
from collections import Counter
from typing import Optional

import jieba

from utils.logger import get_logger

from .base_retriever import BaseRetriever


class BM25Retriever(BaseRetriever):
    index_type = "bm25"

    def __init__(self, k1=1.5, b=0.75, top_k: int = 5):
        self.k1 = float(k1)
        self.b = float(b)
        self.top_k = top_k
        self.logger = get_logger(__name__)

    def _tokenize(self, text: str):
        tokens = list(jieba.cut(text, cut_all=False))
        final_tokens = []
        for token in tokens:
            final_tokens.extend([t for t in re.split(r"\W+", token) if t])
        return final_tokens

    def retrieve(self, index, query_str: str, top_k: Optional[int] = None):
        query_terms = self._tokenize(query_str)
        self.logger.debug(f"query_terms: {query_terms}")
        scores = Counter()
        N = index.N
        avgdl = index.avg_doc_length
        all_doc_ids = set()
        for term in query_terms:
            doc_ids = index.index_data.get(term, [])
            self.logger.debug(f"term '{term}' -> doc_ids: {doc_ids}")
            all_doc_ids.update(doc_ids)
            df = len(doc_ids)
            if df == 0:
                continue
            idf = math.log((N - df + 0.5) / (df + 0.5) + 1)
            for doc_id in doc_ids:
                doc_len = index.doc_lengths[doc_id]
                tf = index.doc_store[doc_id].text.split().count(term)
                score = (
                    idf
                    * (tf * (self.k1 + 1))
                    / (tf + self.k1 * (1 - self.b + self.b * doc_len / avgdl))
                )
                scores[doc_id] += score
        use_top_k = top_k if top_k is not None else self.top_k
        top_docs = [
            index.doc_store[doc_id] for doc_id, _ in scores.most_common(use_top_k)
        ]
        self.logger.debug(f"all matched doc_ids: {list(all_doc_ids)}")
        self.logger.debug(f"top_docs: {[d.text[:50] for d in top_docs]}")
        return top_docs
