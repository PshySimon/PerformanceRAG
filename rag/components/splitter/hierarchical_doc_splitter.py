"""Hierarchical node parser."""

from typing import Any, Dict

from ..base import Component
from .splitter_utils import HierarchicalNodeParser


class HierarchicalDocSplitterComponent(Component):
    """层次分割组件2 - 基于Markdown结构"""

    def __init__(self, name: str, config: Dict[str, Any]):
        super().__init__(name, config)

        # 支持 chunk_sizes 参数
        self.chunk_sizes = config.get("chunk_sizes", [512, 256])
        self.chunk_overlap = config.get("chunk_overlap", 20)
        self.include_metadata = config.get("include_metadata", True)
        self.include_prev_next_rel = config.get("include_prev_next_rel", True)

        # 支持其他配置参数
        self.max_chunk_size = config.get("max_chunk_size", 1024)
        self.fallback_config = config.get("fallback_config", {})

        # 初始化分割器为 None，在 _do_initialize 中创建
        self.splitter = None

    def _do_initialize(self):
        """实现抽象方法：初始化分割器"""
        if self.debug:
            self.logger.debug(
                f"初始化 HierarchicalNodeParser，chunk_sizes: {self.chunk_sizes}"
            )

        self.splitter = HierarchicalNodeParser.from_defaults(
            chunk_sizes=self.chunk_sizes,
            chunk_overlap=self.chunk_overlap,
            include_metadata=self.include_metadata,
            include_prev_next_rel=self.include_prev_next_rel,
        )

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """处理文档数据"""
        if "documents" not in data:
            raise ValueError("输入数据中缺少 'documents' 字段")

        documents = data["documents"]
        if self.debug:
            self.logger.debug(f"开始分割 {len(documents)} 个文档")

        # 使用分割器处理文档
        split_documents = self.splitter.split(documents)

        if self.debug:
            self.logger.debug(f"分割完成，生成 {len(split_documents)} 个文档块")

        return {
            "documents": split_documents,
            "metadata": {
                "component": self.name,
                "chunk_sizes": self.chunk_sizes,
                "total_chunks": len(split_documents),
            },
        }
