from abc import ABC, abstractmethod
from typing import List, Dict, Any


class BaseSplitter(ABC):
    """分割器的抽象基类，定义了分割器的基本接口"""

    @abstractmethod
    def split(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """将文档列表分割成更小的块"""
        pass

    @abstractmethod
    def split_text(self, text: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """将单个文本分割成更小的块"""
        pass