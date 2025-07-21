from abc import ABC, abstractmethod


class BaseRetriever(ABC):
    @abstractmethod
    def retrieve(self, index, query_str: str, top_k: int = 5):
        pass
