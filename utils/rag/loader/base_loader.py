from abc import ABC, abstractmethod
from llama_index.core.schema import Document


class BaseLoader(ABC):
    @abstractmethod
    def load(self) -> list[Document]:
        """加载文档，返回 LlamaIndex 的 Document 列表"""
        pass
