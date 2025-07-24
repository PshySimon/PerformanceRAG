from typing import List
from abc import ABC, abstractmethod

class BaseEmbedding(ABC):
    """嵌入模型基类，定义了嵌入模型的基本接口"""
    
    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        """对单条文本进行embedding"""
        pass
    
    @abstractmethod
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """对多条文本进行批量embedding"""
        pass