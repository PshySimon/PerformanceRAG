from typing import List, Any
from llama_index.core.schema import Document
from llama_index.core.node_parser import HierarchicalNodeParser
from .base_splitter import BaseSplitter

class HierarchicalSplitter(BaseSplitter):
    """
    基于llama_index HierarchicalNodeParser的结构化分割器，支持markdown、html等多种结构化文本。
    """
    def __init__(
        self,
        chunk_sizes=None,
        chunk_overlap=20,
        include_metadata=True,
        include_prev_next_rel=True,
        node_parser_ids=None,
        node_parser_map=None,
        callback_manager=None
    ):
        self.parser = HierarchicalNodeParser.from_defaults(
            chunk_sizes=chunk_sizes,
            chunk_overlap=chunk_overlap,
            include_metadata=include_metadata,
            include_prev_next_rel=include_prev_next_rel,
            node_parser_ids=node_parser_ids,
            node_parser_map=node_parser_map,
            callback_manager=callback_manager
        )

    def split(self, documents: List[Document]) -> List[Any]:
        # 直接返回Node对象列表
        return self.parser.get_nodes_from_documents(documents)

    def split_text(self, text: str) -> List[Any]:
        doc = Document(text=text)
        return self.parser.get_nodes_from_documents([doc]) 