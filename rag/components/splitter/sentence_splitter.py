"""Sentence splitter."""

from typing import Any, Dict, List

from ..base import Component
from .splitter_utils import SentenceSplitter


class SentenceSplitterComponent(Component):
    """句子分割组件"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.chunk_size = config.get("chunk_size", 1024)
        self.chunk_overlap = config.get("chunk_overlap", 200)
        self.separator = config.get("separator", " ")
        self.paragraph_separator = config.get("paragraph_separator", "\n\n\n")
        self.secondary_chunking_regex = config.get(
            "secondary_chunking_regex", "[^,.;。？！]+[,.;。？！]?"
        )
        self.include_metadata = config.get("include_metadata", True)
        self.include_prev_next_rel = config.get("include_prev_next_rel", True)

    def process(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:

        splitter = SentenceSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separator=self.separator,
            paragraph_separator=self.paragraph_separator,
            secondary_chunking_regex=self.secondary_chunking_regex,
            include_metadata=self.include_metadata,
            include_prev_next_rel=self.include_prev_next_rel,
        )
        return splitter.split(documents)
