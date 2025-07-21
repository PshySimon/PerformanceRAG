from .base_reranker import BaseReranker
from typing import List, Any

class EmbeddingReranker(BaseReranker):
    def rerank(self, query: str, nodes: List[Any], top_k: int = 5) -> List[Any]:
        # 简单示例：按内容长度降序排序
        sorted_nodes = sorted(nodes, key=lambda n: len(getattr(n, 'text', getattr(n, 'get_content', lambda: "")())), reverse=True)
        return sorted_nodes[:top_k] 