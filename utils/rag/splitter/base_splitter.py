from abc import ABC, abstractmethod
from typing import List, Any
from llama_index.core.schema import Document


class BaseSplitter(ABC):
    """文档分割器的基类"""
    
    @abstractmethod
    def split(self, documents: List[Document]) -> List[Document]:
        """
        将文档列表分割成更小的chunk
        
        Args:
            documents: 输入的文档列表
            
        Returns:
            分割后的文档列表
        """
        pass
    
    @abstractmethod
    def split_text(self, text: str) -> List[Any]:
        """
        将单个文本分割成chunk列表
        
        Args:
            text: 输入的文本
            
        Returns:
            分割后的文本chunk列表（可为str或Node）
        """
        pass 