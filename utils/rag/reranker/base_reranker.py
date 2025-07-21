from abc import ABC, abstractmethod
from typing import List, Any

class BaseReranker(ABC):
    @abstractmethod
    def rerank(self, query: str, nodes: List[Any], top_k: int = 5) -> List[Any]:
        pass 