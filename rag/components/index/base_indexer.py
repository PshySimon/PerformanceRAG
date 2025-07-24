from abc import abstractmethod
from typing import Any, Dict, List, Optional

from ..base import Component


class BaseIndexer(Component):
    """索引器基类"""

    def __init__(self, name: str, config: Dict[str, Any]):
        super().__init__(name, config)
        self.index_name = config.get("index_name", "default_index")
        self.batch_size = config.get("batch_size", 100)

    @abstractmethod
    def create_index(self, index_name: Optional[str] = None, **kwargs) -> bool:
        """创建索引"""
        pass

    @abstractmethod
    def index_documents(self, documents: List[Dict[str, Any]], index_name: Optional[str] = None) -> bool:
        """索引文档"""
        pass

    @abstractmethod
    def search(self, query: str, top_k: int = 10, index_name: Optional[str] = None, **kwargs) -> List[Dict[str, Any]]:
        """搜索文档"""
        pass

    @abstractmethod
    def get_document(self, doc_id: str, index_name: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """根据ID获取文档"""
        pass

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """处理输入数据"""
        if "documents" in data:
            # 索引文档
            documents = data["documents"]
            success = self.index_documents(documents)

            return {
                "indexed": success,
                "document_count": len(documents),
                "index_name": self.index_name,
                "metadata": {
                    "component": self.name,
                    "indexer_type": self.__class__.__name__,
                },
            }
        elif "query" in data:
            # 搜索文档
            query = data["query"]
            top_k = data.get("top_k", 10)
            results = self.search(query, top_k)

            return {
                "results": results,
                "query": query,
                "result_count": len(results),
                "metadata": {
                    "component": self.name,
                    "indexer_type": self.__class__.__name__,
                },
            }
        else:
            raise ValueError("输入数据必须包含 'documents' 或 'query' 字段")
