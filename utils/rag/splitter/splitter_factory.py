from typing import Dict, Any
from .base_splitter import BaseSplitter
from .text_splitter import TextSplitter
from .recursive_splitter import RecursiveSplitter
from .semantic_splitter import SemanticSplitter
from .hierarchical_splitter import HierarchicalSplitter


def create_splitter(config: Dict[str, Any]) -> BaseSplitter:
    """
    根据配置创建splitter实例
    
    Args:
        config: 配置字典，包含splitter类型和参数
        
    Returns:
        BaseSplitter实例
        
    Example:
        config = {
            "type": "text",
            "chunk_size": 1000,
            "chunk_overlap": 200,
            "split_method": "char"
        }
    """
    splitter_type = config.get("type", "text")
    
    if splitter_type == "text":
        return TextSplitter(
            chunk_size=config.get("chunk_size", 1000),
            chunk_overlap=config.get("chunk_overlap", 200),
            split_method=config.get("split_method", "char"),
            separator=config.get("separator", "\n"),
            keep_separator=config.get("keep_separator", True)
        )
    
    elif splitter_type == "recursive":
        return RecursiveSplitter(
            chunk_size=config.get("chunk_size", 1000),
            chunk_overlap=config.get("chunk_overlap", 200),
            separators=config.get("separators", None),
            keep_separator=config.get("keep_separator", True)
        )
    
    elif splitter_type == "semantic":
        return SemanticSplitter(
            chunk_size=config.get("chunk_size", 1000),
            chunk_overlap=config.get("chunk_overlap", 200),
            similarity_threshold=config.get("similarity_threshold", 0.8),
            min_chunk_size=config.get("min_chunk_size", 100)
        )
    
    elif splitter_type == "hierarchical":
        return HierarchicalSplitter(
            chunk_sizes=config.get("chunk_sizes"),
            chunk_overlap=config.get("chunk_overlap", 20),
            include_metadata=config.get("include_metadata", True),
            include_prev_next_rel=config.get("include_prev_next_rel", True),
            node_parser_ids=config.get("node_parser_ids"),
            node_parser_map=config.get("node_parser_map"),
            callback_manager=config.get("callback_manager")
        )
    else:
        raise ValueError(f"不支持的Splitter类型: {splitter_type}") 