from abc import ABC, abstractmethod
from llama_index.core.schema import Document


class BaseIndex(ABC):
    @abstractmethod
    def build(self, documents: list[Document]) -> bool:
        pass

    @abstractmethod
    def save(self, path: str) -> bool:
        pass

    @classmethod
    @abstractmethod
    def load(cls, path: str) -> "BaseIndex|None":
        pass

    @classmethod
    @abstractmethod
    def is_serialized(cls, path: str) -> bool:
        """判断本地是否已序列化"""
        pass

    def get_search_index(self):
        """返回可用于retriever的索引对象，子类可重载。"""
        return self
