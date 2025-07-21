import logging
import re
from typing import List, Any
from llama_index.core.schema import Document, TextNode
from .base_splitter import BaseSplitter

logger = logging.getLogger(__name__)

class TextSplitter(BaseSplitter):
    """
    基于固定chunk大小的文本分割器
    支持按字符数、单词数或句子数进行分割
    """
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        split_method: str = "char",  # "char", "word", "sentence"
        separator: str = "\n",
        keep_separator: bool = True
    ):
        """
        Args:
            chunk_size: chunk的大小（字符数、单词数或句子数）
            chunk_overlap: 相邻chunk之间的重叠部分
            split_method: 分割方法 ("char", "word", "sentence")
            separator: 分割符
            keep_separator: 是否保留分割符
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.split_method = split_method
        self.separator = separator
        self.keep_separator = keep_separator
        
        if split_method not in ["char", "word", "sentence"]:
            raise ValueError("split_method must be one of: char, word, sentence")
    
    def split(self, documents: List[Document]) -> List[Any]:
        """将文档列表分割成chunk"""
        split_nodes = []
        
        for doc in documents:
            nodes = self.split_text(doc.text)
            
            for node in nodes:
                # 合并原metadata
                node.metadata = {**getattr(doc, "metadata", {}), **getattr(node, "metadata", {})}
                split_nodes.append(node)
        
        return split_nodes
    
    def split_text(self, text: str) -> List[Any]:
        """将文本分割成chunk列表"""
        chunks = []
        
        if self.split_method == "char":
            chunks = self._split_by_char(text)
        elif self.split_method == "word":
            chunks = self._split_by_word(text)
        elif self.split_method == "sentence":
            chunks = self._split_by_sentence(text)
        else:
            raise ValueError(f"Unsupported split method: {self.split_method}")
        
        logger.debug(f"[TextSplitter] split_text: 得到 {len(chunks)} 个chunk, 每个长度: {[len(c) for c in chunks]}")
        # 封装为TextNode
        return [TextNode(text=chunk) for chunk in chunks]
    
    def _split_by_char(self, text: str) -> List[str]:
        """按字符数分割"""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            # 如果不是最后一个chunk，尝试在合适的位置分割
            if end < len(text):
                # 尝试在分隔符处分割
                last_sep = text.rfind(self.separator, start, end)
                if last_sep > start:
                    end = last_sep + (len(self.separator) if self.keep_separator else 0)
            
            chunk = text[start:end]
            chunks.append(chunk)
            
            # 计算下一个chunk的起始位置，考虑重叠
            start = end - self.chunk_overlap
            if start >= len(text):
                break
        
        return chunks
    
    def _split_by_word(self, text: str) -> List[str]:
        """按单词数分割"""
        words = text.split()
        chunks = []
        start = 0
        
        while start < len(words):
            end = start + self.chunk_size
            
            if end > len(words):
                end = len(words)
            
            chunk_words = words[start:end]
            chunk = " ".join(chunk_words)
            chunks.append(chunk)
            
            # 计算下一个chunk的起始位置，考虑重叠
            start = end - (self.chunk_overlap // 2)  # 单词重叠的一半
            if start >= len(words):
                break
        
        return chunks
    
    def _split_by_sentence(self, text: str) -> List[str]:
        """按句子分割"""
        # 简单的句子分割正则表达式
        sentence_pattern = r'[.!?]+'
        sentences = re.split(sentence_pattern, text)
        
        chunks = []
        current_chunk = ""
        sentence_count = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # 如果当前chunk加上新句子会超过限制
            if len(current_chunk) + len(sentence) > self.chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = sentence
                sentence_count = 1
            else:
                current_chunk += " " + sentence if current_chunk else sentence
                sentence_count += 1
        
        # 添加最后一个chunk
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks 