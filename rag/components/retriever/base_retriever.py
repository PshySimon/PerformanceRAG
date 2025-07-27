from abc import abstractmethod
from typing import Any, Dict, List, Optional

from rag.components.base import Component


class BaseRetrieverComponent(Component):
    """检索器组件基类"""

    def __init__(self, name: str, config: Dict[str, Any]):
        super().__init__(name, config)
        self.top_k = config.get("top_k", 10)
        self.similarity_threshold = config.get("similarity_threshold", 0.0)
        self.index_name = config.get("index_name", "default_index")

    @abstractmethod
    def retrieve(
        self, query: str, top_k: Optional[int] = None, **kwargs
    ) -> List[Dict[str, Any]]:
        """检索文档"""
        pass

    @abstractmethod
    def _connect_to_index(self):
        """连接到索引"""
        pass

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """处理输入数据"""
        if "query" not in data:
            raise ValueError("输入数据必须包含 'query' 字段")
    
        query = data["query"]
        top_k = data.get("top_k", self.top_k)
    
        # 执行检索
        results = self.retrieve(query, top_k)
    
        return {
            "documents": results,  # 改为 documents
            "query": query,
            "result_count": len(results),
            "metadata": {
                "component": self.name,
                "retriever_type": self.__class__.__name__,
                "top_k": top_k,
            },
        }
