import logging
from typing import List, Optional, Any
from llama_index.core.schema import Document, TextNode
from .base_splitter import BaseSplitter

logger = logging.getLogger(__name__)

class RecursiveSplitter(BaseSplitter):
    """
    递归分割器
    使用多种分隔符进行层次化分割，从最粗粒度到最细粒度
    """
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: Optional[List[str]] = None,
        keep_separator: bool = True
    ):
        """
        Args:
            chunk_size: chunk的大小
            chunk_overlap: 相邻chunk之间的重叠部分
            separators: 分隔符列表，按优先级排序
            keep_separator: 是否保留分隔符
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]
        self.keep_separator = keep_separator
    
    def split(self, documents: List[Document]) -> List[Any]:
        """将文档列表分割成chunk"""
        split_nodes = []
        
        for doc in documents:
            nodes = self.split_text(doc.text)
            
            for node in nodes:
                node.metadata = {**getattr(doc, "metadata", {}), **getattr(node, "metadata", {})}
                split_nodes.append(node)
        
        return split_nodes
    
    def split_text(self, text: str) -> List[Any]:
        chunks = self._split_recursive(text, self.separators)
        logger.debug(f"[RecursiveSplitter] split_text: 得到 {len(chunks)} 个chunk, 每个长度: {[len(c) for c in chunks]}")
        return [TextNode(text=chunk) for chunk in chunks]
    
    def _split_recursive(self, text: str, separators: List[str]) -> List[str]:
        if len(text) <= self.chunk_size:
            logger.debug(f"[RecursiveSplitter] _split_recursive: 文本长度{len(text)} <= chunk_size{self.chunk_size}, 返回1块")
            return [text]
        
        # 尝试使用当前分隔符分割
        separator = separators[0]
        remaining_separators = separators[1:]
        
        if separator == "":
            logger.debug(f"[RecursiveSplitter] _split_recursive: 用字符分割, 文本长度{len(text)}")
            # 如果没有找到合适的分隔符，直接按字符分割
            return self._split_by_char(text)
        
        splits = text.split(separator)
        logger.debug(f"[RecursiveSplitter] _split_recursive: 用分隔符'{separator}'分割, 得到{len(splits)}块")
        
        # 如果分割后每个部分都小于chunk_size，则合并
        if all(len(split) <= self.chunk_size for split in splits):
            chunks = []
            current_chunk = ""
            
            for split in splits:
                if current_chunk and len(current_chunk + separator + split) <= self.chunk_size:
                    current_chunk += separator + split if self.keep_separator else split
                else:
                    if current_chunk:
                        chunks.append(current_chunk)
                    current_chunk = split
            
            if current_chunk:
                chunks.append(current_chunk)
            logger.debug(f"[RecursiveSplitter] _split_recursive: 合并后得到{len(chunks)}块")
            return chunks
        else:
            chunks = []
            # 递归分割每个部分
            for split in splits:
                if len(split) <= self.chunk_size:
                    chunks.append(split)
                else:
                    sub_chunks = self._split_recursive(split, remaining_separators)
                    chunks.extend(sub_chunks)
            logger.debug(f"[RecursiveSplitter] _split_recursive: 递归后得到{len(chunks)}块")
            return chunks
    
    def _split_by_char(self, text: str) -> List[str]:
        """按字符数分割"""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            
            start = end - self.chunk_overlap
            if start >= len(text):
                break
        
        return chunks 