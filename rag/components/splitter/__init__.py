from ...pipeline.registry import ComponentRegistry
from .hierarchical_splitter import HierarchicalSplitterComponent
from .recursive_splitter import RecursiveSplitterComponent
from .semantic_splitter import SemanticSplitterComponent
from .text_splitter import TextSplitterComponent
from .hierarchical_doc_splitter import HierarchicalDocSplitterComponent
from .sentence_splitter import SentenceSplitterComponent
# 导出分割器工具类
from .base_splitter import BaseSplitter
from .enums import SplitMethod, SplitterType
from .splitter_factory import create_splitter
from .splitter_utils import (
    HierarchicalSplitter,
    RecursiveSplitter,
    SemanticSplitter,
    TextSplitter,
    HierarchicalNodeParser,
    SentenceSplitter
)

# 注册组件
ComponentRegistry.register("splitter", "text")(TextSplitterComponent)
ComponentRegistry.register("splitter", "recursive")(RecursiveSplitterComponent)
ComponentRegistry.register("splitter", "semantic")(SemanticSplitterComponent)
ComponentRegistry.register("splitter", "hierarchical")(HierarchicalSplitterComponent)
ComponentRegistry.register("splitter", "hierarchical_doc")(HierarchicalDocSplitterComponent)
ComponentRegistry.register("splitter", "sentence")(SentenceSplitterComponent)


__all__ = [
    "TextSplitterComponent",
    "RecursiveSplitterComponent",
    "SemanticSplitterComponent",
    "HierarchicalSplitterComponent",
    "HierarchicalDocSplitterComponent",
    "SentenceSplitterComponent",
    "BaseSplitter",
    "TextSplitter",
    "RecursiveSplitter",
    "SemanticSplitter",
    "HierarchicalSplitter",
    "create_splitter",
    "SplitMethod",
    "SplitterType",
    "HierarchicalNodeParser",
    "SentenceSplitter"
]
