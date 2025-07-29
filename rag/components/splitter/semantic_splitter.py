from typing import Dict, Any, List

from ..base import Component
from .splitter_utils import SemanticSplitter


class SemanticSplitterComponent(Component):
    """语义分割组件，使用SemanticSplitter进行文档分割"""

    def __init__(self, name: str, config: Dict[str, Any]):
        """初始化语义分割组件

        Args:
            name: 组件名称
            config: 组件配置
        """
        super().__init__(name, config)
        
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
        
        # 初始化时不创建splitter，在_do_initialize中创建
        self.splitter = None

    def _do_initialize(self):
        """实际的初始化逻辑"""
        if self.debug:
            self.logger.debug(f"初始化语义分割器，chunk_size={self.chunk_size}, similarity_threshold={self.similarity_threshold}")
        
        self.splitter = SemanticSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            similarity_threshold=self.similarity_threshold,
            min_chunk_size=self.min_chunk_size,
            embedding_model=self.embedding_model,
            include_metadata=self.include_metadata,
            max_chunk_size=self.max_chunk_size,
        )

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """处理文档列表，将其分割成更小的块

        Args:
            data: 包含documents字段的数据字典

        Returns:
            Dict[str, Any]: 包含分割后文档列表的数据字典
        """
        if self.splitter is None:
            raise RuntimeError("组件未初始化，请先调用initialize()方法")
        
        documents = data.get("documents", [])
        if self.debug:
            self.logger.debug(f"开始分割 {len(documents)} 个文档")
        
        split_documents = self.splitter.split(documents)
        
        if self.debug:
            self.logger.debug(f"分割完成，生成 {len(split_documents)} 个文档块")
        
        return {"documents": split_documents}