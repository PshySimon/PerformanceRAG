from typing import Any, Dict

from ..base import Component
from .splitter_utils import RecursiveSplitter


class RecursiveSplitterComponent(Component):
    """递归分割组件，使用RecursiveSplitter进行文档分割"""

    def __init__(self, name: str, config: Dict[str, Any]):
        """初始化递归分割组件

        Args:
            name: 组件名称
            config: 组件配置
        """
        super().__init__(name, config)

        try:
            self.chunk_size = config["chunk_size"]
        except KeyError:
            raise ValueError("RecursiveSplitterComponent 必须指定 chunk_size 参数")

        self.chunk_overlap = config["chunk_overlap"]
        self.separators = config["separators"]
        self.keep_separator = config["keep_separator"]
        self.include_metadata = config["include_metadata"]
        self.splitter = None

    def _do_initialize(self):
        """实际的初始化逻辑"""
        self.splitter = RecursiveSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators,
            keep_separator=self.keep_separator,
            include_metadata=self.include_metadata,
        )
        if self.debug:
            self.logger.debug(
                f"RecursiveSplitter 初始化完成，chunk_size={self.chunk_size}"
            )

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """处理输入数据

        Args:
            data: 输入数据，包含documents字段

        Returns:
            Dict[str, Any]: 处理结果，包含documents字段
        """
        documents = data.get("documents", [])
        if not documents:
            return {"documents": []}

        result_documents = self.splitter.split(documents)
        return {"documents": result_documents}
