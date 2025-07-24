from typing import Dict, Any, List

from ..base import Component
from .splitter_utils import TextSplitter
from .enums import SplitMethod


class TextSplitterComponent(Component):
    """文本分割组件，使用TextSplitter进行文档分割"""

    def __init__(self, config: Dict[str, Any]):
        """初始化文本分割组件

        Args:
            config: 组件配置
        """
        super().__init__(config)
        self.chunk_size = config["chunk_size"]

        self.chunk_overlap = config["chunk_overlap"] if "chunk_overlap" in config else 0
        self.include_metadata = config["include_metadata"] if "include_metadata" in config else True
        
        # 处理枚举参数
        try:
            split_method_str = config["split_method"] if "split_method" in config else "character"
            self.split_method = SplitMethod.from_str(split_method_str)
        except ValueError as e:
            raise ValueError(str(e))

    def process(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """处理文档列表，将其分割成更小的块

        Args:
            documents: 文档列表，每个文档是一个字典，包含content和metadata字段

        Returns:
            List[Dict[str, Any]]: 分割后的文档列表
        """
        splitter = TextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            split_method=self.split_method,
            include_metadata=self.include_metadata,
        )
        return splitter.split(documents)