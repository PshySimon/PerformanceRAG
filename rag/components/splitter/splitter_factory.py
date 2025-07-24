from typing import Any, Dict

from .base_splitter import BaseSplitter
from .enums import SplitMethod, SplitterType
from .splitter_utils import (
    HierarchicalNodeParser,
    HierarchicalSplitter,
    RecursiveSplitter,
    SemanticSplitter,
    SentenceSplitter,  # 新增导入
    TextSplitter,
)


def create_splitter(config: Dict[str, Any]) -> BaseSplitter:
    """创建分割器实例

    Args:
        config: 分割器配置

    Returns:
        BaseSplitter: 分割器实例
    """
    splitter_type = config["type"]
    splitter_type_enum = SplitterType.from_str(splitter_type)

    if splitter_type_enum == SplitterType.TEXT:
        chunk_size = config["chunk_size"]
        chunk_overlap = config["chunk_overlap"]
        include_metadata = config["include_metadata"]
        split_method_str = config["split_method"]
        split_method = SplitMethod.from_str(split_method_str)

        return TextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            split_method=split_method,
            include_metadata=include_metadata,
        )

    elif splitter_type_enum == SplitterType.RECURSIVE:
        chunk_size = config["chunk_size"]
        chunk_overlap = config["chunk_overlap"]
        separators = config["separators"]
        keep_separator = config["keep_separator"]
        include_metadata = config["include_metadata"]

        return RecursiveSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
            keep_separator=keep_separator,
            include_metadata=include_metadata,
        )

    elif splitter_type_enum == SplitterType.SEMANTIC:
        chunk_size = config["chunk_size"]
        similarity_threshold = config["similarity_threshold"]
        embedding_model = config["embedding_model"]
        chunk_overlap = config["chunk_overlap"]
        min_chunk_size = config["min_chunk_size"]
        include_metadata = config["include_metadata"]

        return SemanticSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            similarity_threshold=similarity_threshold,
            min_chunk_size=min_chunk_size,
            embedding_model=embedding_model,
            include_metadata=include_metadata,
        )

    elif splitter_type_enum == SplitterType.HIERARCHICAL:
        chunk_sizes = config["chunk_sizes"]
        chunk_overlap = config["chunk_overlap"]
        include_metadata = config["include_metadata"]
        # 添加对 splitter_type 参数的支持，默认为 "text"
        splitter_type = config.get("splitter_type", "text")

        return HierarchicalSplitter(
            chunk_sizes=chunk_sizes,
            chunk_overlap=chunk_overlap,
            include_metadata=include_metadata,
            splitter_type=splitter_type,
        )
    elif splitter_type_enum == SplitterType.SENTENCE:
        chunk_size = config.get("chunk_size", 1024)
        chunk_overlap = config.get("chunk_overlap", 200)
        separator = config.get("separator", " ")
        paragraph_separator = config.get("paragraph_separator", "\n\n\n")
        secondary_chunking_regex = config.get(
            "secondary_chunking_regex", "[^,.;。？！]+[,.;。？！]?"
        )
        include_metadata = config.get("include_metadata", True)
        include_prev_next_rel = config.get("include_prev_next_rel", True)

        return SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separator=separator,
            paragraph_separator=paragraph_separator,
            secondary_chunking_regex=secondary_chunking_regex,
            include_metadata=include_metadata,
            include_prev_next_rel=include_prev_next_rel,
        )
    elif splitter_type_enum == SplitterType.HIERARCHICAL_DOC:
        chunk_size = config.get("chunk_size", 1024)
        chunk_overlap = config.get("chunk_overlap", 200)
        include_metadata = config.get("include_metadata", True)
        include_prev_next_rel = config.get("include_prev_next_rel", True)

        return HierarchicalNodeParser(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            include_metadata=include_metadata,
            include_prev_next_rel=include_prev_next_rel,
        )
    else:
        raise ValueError(f"不支持的Splitter类型: {splitter_type}")
