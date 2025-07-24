from typing import Dict, Any, List

from ..base import Component
from .splitter_utils import SemanticSplitter


class SemanticSplitterComponent(Component):
    """语义分割组件，使用SemanticSplitter进行文档分割"""

    def __init__(self, config: Dict[str, Any]):
        """初始化语义分割组件

        Args:
            config: 组件配置
        """
        super().__init__(config)
        
        # 直接访问必要参数
        self.chunk_size = config["chunk_size"]
        self.similarity_threshold = config["similarity_threshold"]
        self.embedding_model = config["embedding_model"]
    
        # 处理可选参数
        self.chunk_overlap = config["chunk_overlap"] 
        self.min_chunk_size = config["min_chunk_size"] 
        self.include_metadata = config["include_metadata"]
        # 添加max_chunk_size参数
        self.max_chunk_size = config.get("max_chunk_size", None)

    def process(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """处理文档列表，将其分割成更小的块

        Args:
            documents: 文档列表，每个文档是一个字典，包含content和metadata字段

        Returns:
            List[Dict[str, Any]]: 分割后的文档列表
        """
        splitter = SemanticSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            similarity_threshold=self.similarity_threshold,
            min_chunk_size=self.min_chunk_size,
            embedding_model=self.embedding_model,
            include_metadata=self.include_metadata,
            max_chunk_size=self.max_chunk_size,  # 传递max_chunk_size参数
        )
        return splitter.split(documents)