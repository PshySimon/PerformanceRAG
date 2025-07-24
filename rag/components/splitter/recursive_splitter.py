from typing import Dict, Any, List

from ..base import Component
from .splitter_utils import RecursiveSplitter


class RecursiveSplitterComponent(Component):
    """递归分割组件，使用RecursiveSplitter进行文档分割"""

    def __init__(self, config: Dict[str, Any]):
        """初始化递归分割组件

        Args:
            config: 组件配置
        """
        super().__init__(config)
        
        try:
            self.chunk_size = config["chunk_size"]
        except KeyError:
            raise ValueError("RecursiveSplitterComponent 必须指定 chunk_size 参数")
            
        self.chunk_overlap = config["chunk_overlap"]
        self.separators = config["separators"]
        self.keep_separator = config["keep_separator"]
        self.include_metadata = config["include_metadata"]

    def process(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """处理文档列表，将其分割成更小的块

        Args:
            documents: 文档列表，每个文档是一个字典，包含content和metadata字段

        Returns:
            List[Dict[str, Any]]: 分割后的文档列表
        """
        splitter = RecursiveSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators,
            keep_separator=self.keep_separator,
            include_metadata=self.include_metadata,
        )
        return splitter.split(documents)