import logging
from typing import Any, List, Optional

from llama_index.core.schema import Document, TextNode

from utils.config import config
from utils.rag.embedding.base import BaseEmbedding
from utils.rag.embedding.factory import EmbeddingFactory

from .base_splitter import BaseSplitter

logger = logging.getLogger(__name__)


class SemanticSplitter(BaseSplitter):
    """
    语义分割器
    基于语义相似度进行分割，保持语义完整性
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        similarity_threshold: float = 0.8,
        min_chunk_size: int = 100,
    ):
        """
        Args:
            chunk_size: chunk的目标大小
            chunk_overlap: 相邻chunk之间的重叠部分
            similarity_threshold: 语义相似度阈值
            min_chunk_size: 最小chunk大小
        """
        embedding_client = EmbeddingFactory.from_config()
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._embedding_client = embedding_client
        self.similarity_threshold = similarity_threshold
        self.min_chunk_size = min_chunk_size

    def _init_embedding_client(self):
        pass  # 工厂模式后不再需要此方法

    def split(self, documents: List[Document]) -> List[Any]:
        """将文档列表分割成chunk"""
        split_nodes = []

        for doc in documents:
            nodes = self.split_text(doc.text)

            for node in nodes:
                node.metadata = {
                    **getattr(doc, "metadata", {}),
                    **getattr(node, "metadata", {}),
                }
                split_nodes.append(node)

        return split_nodes

    def split_text(self, text: str) -> List[Any]:
        """基于语义分割文本"""
        chunks = self._semantic_split(text)
        return [TextNode(text=chunk) for chunk in chunks]

    def _semantic_split(self, text: str) -> List[str]:
        """基于语义相似度分割文本"""
        # 首先按句子分割
        sentences = self._split_into_sentences(text)

        if len(sentences) <= 1:
            return [text]

        # 进度条支持
        try:
            from tqdm import tqdm

            iterator = tqdm(sentences, desc="[SemanticSplitter] 分割进度", ncols=80)
        except ImportError:
            iterator = sentences  # 直接用原始list，无需定义tqdm函数

        chunks = []
        current_chunk = ""
        current_sentences = []

        for idx, sentence in enumerate(iterator):
            # 如果当前chunk为空，直接添加句子
            if not current_chunk:
                current_chunk = sentence
                current_sentences = [sentence]
                continue

            # 计算添加新句子后的语义相似度
            test_chunk = current_chunk + " " + sentence

            if len(test_chunk) <= self.chunk_size:
                # 检查语义相似度
                if self._check_semantic_similarity(current_sentences, sentence):
                    current_chunk = test_chunk
                    current_sentences.append(sentence)
                else:
                    # 语义不相似，开始新的chunk
                    chunks.append(current_chunk)
                    current_chunk = sentence
                    current_sentences = [sentence]
            else:
                # 超过大小限制，开始新的chunk
                chunks.append(current_chunk)
                current_chunk = sentence
                current_sentences = [sentence]

        # 添加最后一个chunk
        if current_chunk:
            chunks.append(current_chunk)

        logger.debug(f"[SemanticSplitter] _semantic_split: 得到 {len(chunks)} 个chunk")
        return chunks

    def _split_into_sentences(self, text: str) -> List[str]:
        """将文本分割成句子"""
        import re

        # 简单的句子分割
        sentence_pattern = r"[.!?。！？]+"
        sentences = re.split(sentence_pattern, text)

        # 清理和过滤
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences

    def _check_semantic_similarity(
        self, current_sentences: List[str], new_sentence: str
    ) -> bool:
        """检查语义相似度"""
        if not self._embedding_client:
            return self._heuristic_similarity(current_sentences, new_sentence)
        try:
            current_text = " ".join(current_sentences)
            current_embedding = self._embedding_client.embed_text(current_text)
            new_embedding = self._embedding_client.embed_text(new_sentence)
            similarity = self._cosine_similarity(current_embedding, new_embedding)
            return similarity >= self.similarity_threshold
        except Exception as e:
            logger.warning(
                f"Embedding similarity calculation failed: {e}, falling back to heuristic"
            )
            return self._heuristic_similarity(current_sentences, new_sentence)

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """计算两个向量的余弦相似度"""
        import math

        if len(vec1) != len(vec2):
            return 0.0

        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def _heuristic_similarity(
        self, current_sentences: List[str], new_sentence: str
    ) -> bool:
        """启发式相似度检查"""
        # 简单的启发式方法：检查关键词重叠
        if not current_sentences:
            return True

        # 获取当前chunk的关键词
        current_text = " ".join(current_sentences)
        current_words = set(current_text.lower().split())

        # 获取新句子的关键词
        new_words = set(new_sentence.lower().split())

        # 计算重叠度
        overlap = len(current_words.intersection(new_words))
        total_unique = len(current_words.union(new_words))

        if total_unique == 0:
            return True

        similarity = overlap / total_unique
        return similarity >= self.similarity_threshold

    # 移除_fallback_split方法
