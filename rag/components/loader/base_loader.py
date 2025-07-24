from abc import ABC, abstractmethod
from typing import List, Dict, Any


class BaseLoader(ABC):
    """加载器的抽象基类，定义了加载器的基本接口"""

    @abstractmethod
    def load(self) -> List[Dict[str, Any]]:
        """加载文档并返回文档列表"""
        pass