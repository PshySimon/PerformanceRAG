from abc import ABC, abstractmethod
from typing import List


class BaseEmbedding(ABC):
    """嵌入模型基类，定义了嵌入模型的基本接口"""

    def __init__(self, dimensions: int = None):
        """初始化基础embedding类

        Args:
            dimensions: 嵌入向量维度
        """
        self.dimensions = dimensions

    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        """对单条文本进行embedding"""
        pass

    @abstractmethod
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """对多条文本进行批量embedding"""
        pass

    def get_dimensions(self) -> int:
        """获取嵌入向量维度"""
        return self.dimensions
