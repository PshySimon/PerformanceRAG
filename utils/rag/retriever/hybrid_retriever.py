from typing import Any, List, Optional

import numpy as np

from .base_retriever import BaseRetriever

class HybridRetriever(BaseRetriever):
    def __init__(
        self,
        strategy: str = "weighted",
        normalization: str = "minmax",
        weights: Optional[List[float]] = None,
        retriever_objs: Optional[List[Any]] = None,
        top_k: int = 5,
    ):
        self.strategy = strategy
        self.normalization = normalization
        self.weights = weights or [1.0] * (len(retriever_objs) if retriever_objs else 1)
        self.top_k = top_k
        self.retriever_objs = retriever_objs or []

    def _normalize(self, scores: List[float]):
        if not scores:
            return []
        if self.normalization == "minmax":
            min_s, max_s = min(scores), max(scores)
            if max_s == min_s:
                return [1.0 for _ in scores]
            return [(s - min_s) / (max_s - min_s) for s in scores]
        elif self.normalization == "zscore":
            mean, std = np.mean(scores), np.std(scores)
            if std == 0:
                return [1.0 for _ in scores]
            return [(s - mean) / std for s in scores]
        elif self.normalization == "softmax":
            exp_scores = np.exp(scores - np.max(scores))
            return (exp_scores / exp_scores.sum()).tolist()
        else:
            return scores

    def retrieve(self, index, query_str: str, top_k: Optional[int] = None):
        top_k = top_k or self.top_k
        all_results = []
        all_scores = []
        for i, retriever in enumerate(self.retriever_objs):
            # 支持index为dict，自动分发
            if isinstance(index, dict):
                idx_type = getattr(retriever, "index_type", None)
                sub_index = index.get(idx_type)
            else:
                sub_index = index
            results = retriever.retrieve(sub_index, query_str, top_k=top_k)
            all_results.append(results)
            scores = [getattr(n, "score", 1.0 / (j + 1)) for j, n in enumerate(results)]
            all_scores.append(scores)
        # 扁平化并去重
        node_to_scores = {}
        for idx, results in enumerate(all_results):
            norm_scores = self._normalize(all_scores[idx])
            for i, node in enumerate(results):
                node_id = getattr(node, "node_id", id(node))
                if node_id not in node_to_scores:
                    node_to_scores[node_id] = {
                        "node": node,
                        "scores": [0.0] * len(self.retriever_objs),
                    }
                node_to_scores[node_id]["scores"][idx] = norm_scores[i]
        # 融合
        if self.strategy == "weighted":
            for v in node_to_scores.values():
                v["final_score"] = sum(
                    [w * s for w, s in zip(self.weights, v["scores"])]
                )
            sorted_nodes = sorted(
                node_to_scores.values(), key=lambda x: x["final_score"], reverse=True
            )
        elif self.strategy == "rrf":
            for v in node_to_scores.values():
                v["rrf_score"] = sum(
                    [
                        1.0 / (50 * rank + 1) if s > 0 else 0
                        for rank, s in enumerate(v["scores"])
                    ]
                )
            sorted_nodes = sorted(
                node_to_scores.values(), key=lambda x: x["rrf_score"], reverse=True
            )
        else:
            raise ValueError(f"不支持的Hybrid策略: {self.strategy}")
        return [x["node"] for x in sorted_nodes[:top_k]]
