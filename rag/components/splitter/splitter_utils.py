from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple

from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.callbacks.schema import CBEventType, EventPayload
from llama_index.core.constants import DEFAULT_CHUNK_SIZE
from llama_index.core.node_parser.interface import MetadataAwareTextSplitter, NodeParser
from llama_index.core.node_parser.node_utils import default_id_func
from llama_index.core.node_parser.text.utils import (
    split_by_char,
    split_by_regex,
    split_by_sentence_tokenizer,
    split_by_sep,
)
from llama_index.core.schema import BaseNode, Document, NodeRelationship
from llama_index.core.utils import get_tokenizer

from utils.logger import get_logger

from ..embedding.embedding_factory import EmbeddingFactory
from .base_splitter import BaseSplitter
from .enums import SplitMethod

SENTENCE_CHUNK_OVERLAP = 200
CHUNKING_REGEX = "[^,.;。？！]+[,.;。？！]?"
DEFAULT_PARAGRAPH_SEP = "\n\n\n"


@dataclass
class _Split:
    text: str  # the split text
    is_sentence: bool  # save whether this is a full sentence
    token_size: int  # token length of split text


logger = get_logger(__name__)


# 修改 TextSplitter 类的 __init__ 方法
class TextSplitter(BaseSplitter):
    """基于固定chunk大小的文本分割器"""

    def __init__(
        self,
        chunk_size: int,  # 移除默认值，作为必选参数
        chunk_overlap: int,  # 移除默认值，作为必选参数
        split_method: SplitMethod,  # 使用枚举类型
        include_metadata: bool,  # 移除默认值
    ):
        """初始化文本分割器

        Args:
            chunk_size: 每个chunk的最大大小
            chunk_overlap: 相邻chunk之间的重叠大小
            split_method: 分割方法，支持 'character', 'word', 'sentence'
            include_metadata: 是否在分割后的chunk中包含原始文档的元数据
        """
        if chunk_size <= 0:
            raise ValueError("chunk_size必须大于0")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap必须大于等于0")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap必须小于chunk_size")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.split_method = split_method
        self.include_metadata = include_metadata

    # 修改 split_text 方法中的分割方法判断
    def split_text(
        self, text: str, metadata: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """将文本分割成更小的块"""
        if not text:
            return []

        metadata = metadata or {}

        # 根据分割方法选择不同的分割函数
        if self.split_method == SplitMethod.CHARACTER:
            chunks = self._split_by_character(text)
        elif self.split_method == SplitMethod.WORD:
            chunks = self._split_by_word(text)
        elif self.split_method == SplitMethod.SENTENCE:
            chunks = self._split_by_sentence(text)
        else:
            # 默认使用字符分割作为退化方式
            chunks = self._split_by_character(text)

        # 将分割后的文本转换为字典格式
        result = []
        for i, chunk in enumerate(chunks):
            chunk_metadata = metadata.copy() if self.include_metadata else {}
            chunk_metadata["chunk_index"] = i
            result.append({"content": chunk, "metadata": chunk_metadata})

        return result

    def split(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """将文档列表分割成更小的块"""
        if not documents:
            return []

        result = []
        # 移除进度条，直接遍历文档
        for doc in documents:
            content = doc.get("content", "")
            metadata = doc.get("metadata", {})
            if not content:
                continue

            chunks = self.split_text(content, metadata)
            result.extend(chunks)

        return result

    def split_stream(self, documents: List[Dict[str, Any]]) -> Iterator[Dict[str, Any]]:
        """流式分割文档"""
        if not documents:
            return

        for doc in documents:
            content = doc.get("content", "")
            metadata = doc.get("metadata", {})
            if not content:
                continue

            # 逐个分割并yield每个chunk
            chunks = self.split_text(content, metadata)
            for chunk in chunks:
                yield chunk

    def _split_by_character(self, text: str) -> List[str]:
        """按字符数分割"""
        chunks = []
        start = 0

        while start < len(text):
            end = start + self.chunk_size

            # 如果不是最后一个chunk，尝试在合适的位置分割
            if end < len(text):
                # 尝试在换行符处分割
                newline_pos = text.rfind("\n", start, end)
                if newline_pos != -1:
                    end = newline_pos + 1
                # 如果没有换行符，尝试在空格处分割
                elif " " in text[start:end]:
                    space_pos = text.rfind(" ", start, end)
                    if space_pos != -1:
                        end = space_pos + 1

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            # 计算下一个chunk的起始位置（考虑重叠）
            start = max(start + 1, end - self.chunk_overlap)

        return chunks

    def _split_by_word(self, text: str) -> List[str]:
        """按单词数分割"""
        words = text.split()
        chunks = []
        start = 0

        while start < len(words):
            end = start + self.chunk_size
            chunk_words = words[start:end]
            chunk = " ".join(chunk_words)

            if chunk:
                chunks.append(chunk)

            # 计算下一个chunk的起始位置（考虑重叠）
            start = max(start + 1, end - self.chunk_overlap)

        return chunks

    def _split_by_sentence(self, text: str) -> List[str]:
        """按句子分割"""
        import re

        # 简单的句子分割正则表达式
        sentence_pattern = r"[.!?。！？]+"
        sentences = re.split(sentence_pattern, text)

        chunks = []
        current_chunk = ""

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            # 如果当前chunk加上新句子会超过限制
            if len(current_chunk) + len(sentence) > self.chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                current_chunk += " " + sentence if current_chunk else sentence

        # 添加最后一个chunk
        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks


# 修改 RecursiveSplitter 类的 __init__ 方法
class RecursiveSplitter(BaseSplitter):
    """递归分割器，使用多个分隔符递归分割文本"""

    def __init__(
        self,
        chunk_size: int,  # 移除默认值，作为必选参数
        chunk_overlap: int,  # 移除默认值，作为必选参数
        separators: List[str],  # 移除默认值，作为必选参数
        keep_separator: bool,  # 移除默认值，作为必选参数
        include_metadata: bool,  # 移除默认值，作为必选参数
    ):
        """初始化递归分割器

        Args:
            chunk_size: 每个chunk的最大大小
            chunk_overlap: 相邻chunk之间的重叠大小
            separators: 分隔符列表，按优先级排序
            keep_separator: 是否在分割后的chunk中保留分隔符
            include_metadata: 是否在分割后的chunk中包含原始文档的元数据
        """
        if chunk_size <= 0:
            raise ValueError("chunk_size必须大于0")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap必须大于等于0")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap必须小于chunk_size")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        # 如果没有提供分隔符，使用默认分隔符
        self.separators = separators if separators else ["\n\n", "\n", " ", ""]
        self.keep_separator = keep_separator
        self.include_metadata = include_metadata


class SemanticSplitter(BaseSplitter):
    """语义分割器，基于语义相似度分割文本"""

    def __init__(
        self,
        chunk_size: int,
        chunk_overlap: int,
        similarity_threshold: float,
        min_chunk_size: int,
        embedding_model: str,
        include_metadata: bool,
        max_chunk_size: int = None,
        **kwargs,  # 新增：接收所有额外的嵌入相关参数
    ):
        """初始化语义分割器

        Args:
            chunk_size: 每个chunk的最大大小
            chunk_overlap: 相邻chunk之间的重叠大小
            similarity_threshold: 语义相似度阈值，超过该阈值的句子将被合并到同一个chunk
            min_chunk_size: 最小chunk大小，小于该大小的chunk将被合并
            embedding_model: 嵌入模型名称
            include_metadata: 是否在分割后的chunk中包含原始文档的元数据
            max_chunk_size: 最大chunk大小阈值，超过该阈值的文本将使用退化方法分割，默认为chunk_size的2倍
            **kwargs: 其他嵌入相关参数，如api_key、api_base等
        """
        if chunk_size <= 0:
            raise ValueError("chunk_size必须大于0")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap必须大于等于0")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap必须小于chunk_size")
        if similarity_threshold < 0 or similarity_threshold > 1:
            raise ValueError("similarity_threshold必须在0到1之间")
        if min_chunk_size <= 0:
            raise ValueError("min_chunk_size必须大于0")
        if min_chunk_size >= chunk_size:
            raise ValueError("min_chunk_size必须小于chunk_size")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.similarity_threshold = similarity_threshold
        self.min_chunk_size = min_chunk_size
        self.embedding_model = embedding_model
        self.include_metadata = include_metadata
        # 如果未提供max_chunk_size，则默认为chunk_size的2倍
        self.max_chunk_size = (
            max_chunk_size if max_chunk_size is not None else chunk_size * 2
        )

        # 初始化嵌入客户端 - 使用新的嵌入组件
        try:
            # 传递所有嵌入相关参数
            self.embedding_client = EmbeddingFactory.create(
                "openai", model=embedding_model, **kwargs
            )
            logger.info(f"成功初始化OpenAI嵌入客户端，模型: {embedding_model}")
        except Exception as e:
            logger.warning(f"初始化嵌入客户端失败: {e}，将使用启发式方法进行分割")
            self.embedding_client = None

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
        if not self.embedding_client:
            return self._heuristic_similarity(current_sentences, new_sentence)
        # 在 _check_semantic_similarity 方法中
        try:
            current_text = " ".join(current_sentences)
            current_embedding = self.embedding_client.embed_text(current_text)
            new_embedding = self.embedding_client.embed_text(new_sentence)
            similarity = self._cosine_similarity(current_embedding, new_embedding)
            return similarity >= self.similarity_threshold
        except Exception as e:
            logger.warning(f"嵌入相似度计算失败: {e}，将使用启发式方法")
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

    def _semantic_split(self, text: str) -> List[str]:
        """基于语义相似度分割文本"""
        # 首先按句子分割文本
        sentences = self._split_into_sentences(text)

        # 如果总长度小于chunk_size，直接返回整个文本
        if sum(len(s) for s in sentences) <= self.chunk_size:
            return [text]

        # 如果句子数量过多或单个句子长度超过max_chunk_size，退化为按字符分割
        if (
            len(sentences) > 1000
            or max(len(s) for s in sentences) > self.max_chunk_size
        ):
            # 退化为使用TextSplitter的字符分割方法
            text_splitter = TextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                split_method=SplitMethod.CHARACTER,
                include_metadata=self.include_metadata,
            )
            chunks = text_splitter._split_by_character(text)
            return chunks

        # 原有的语义分割逻辑
        if len(sentences) <= 1:
            return [text]

        chunks = []
        current_chunk = ""
        current_sentences = []

        for idx, sentence in enumerate(sentences):
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

        return chunks

    def split(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """将文档列表分割成更小的块"""
        if not documents:
            return []

        result = []
        # 移除进度条，直接遍历文档
        for doc in documents:
            content = doc.get("content", "")
            metadata = doc.get("metadata", {})
            if not content:
                continue

            chunks = self.split_text(content, metadata)
            result.extend(chunks)

        return result

    def split_text(
        self, text: str, metadata: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """将文本分割成更小的块"""
        if not text:
            return []

        metadata = metadata or {}

        # 使用语义分割方法
        chunks = self._semantic_split(text)

        # 将分割后的文本转换为字典格式
        result = []
        for i, chunk in enumerate(chunks):
            chunk_metadata = metadata.copy() if self.include_metadata else {}
            chunk_metadata["chunk_index"] = i
            chunk_metadata["split_method"] = "semantic"
            result.append({"content": chunk, "metadata": chunk_metadata})

        return result


# 修改 HierarchicalSplitter 类的 __init__ 方法
class HierarchicalSplitter(BaseSplitter):
    """层次分割器，基于llama_index的HierarchicalNodeParser实现"""

    def __init__(
        self,
        chunk_sizes: List[int],
        chunk_overlap: int,
        include_metadata: bool,
        max_chunk_size: int = None,
        fallback_config: Dict[str, Any] = None,  # 新增fallback配置
    ):
        """初始化层次分割器

        Args:
            chunk_sizes: 每个层次的chunk大小列表
            chunk_overlap: 相邻chunk之间的重叠大小
            include_metadata: 是否在分割后的chunk中包含原始文档的元数据
            max_chunk_size: 最大chunk大小阈值
            fallback_config: 退化分割器配置
        """
        if not chunk_sizes or not isinstance(chunk_sizes, list):
            raise ValueError("chunk_sizes必须是非空列表")
        if any(size <= 0 for size in chunk_sizes):
            raise ValueError("chunk_sizes中的所有值必须大于0")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap必须大于等于0")
        if chunk_overlap >= min(chunk_sizes):
            raise ValueError("chunk_overlap必须小于最小的chunk_size")

        self.chunk_sizes = chunk_sizes
        self.chunk_overlap = chunk_overlap
        self.include_metadata = include_metadata
        self.largest_chunk_size = max(chunk_sizes)
        self.max_chunk_size = (
            max_chunk_size
            if max_chunk_size is not None
            else self.largest_chunk_size * 2
        )

        # 处理fallback配置
        self.fallback_config = self._process_fallback_config(fallback_config or {})

        # 创建HierarchicalNodeParser
        self.parser = HierarchicalNodeParser.from_defaults(
            chunk_sizes=self.chunk_sizes, chunk_overlap=self.chunk_overlap
        )

    def _process_fallback_config(
        self, fallback_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """处理和验证fallback配置"""
        # 如果用户没有提供fallback配置，使用默认的单层TextSplitter兜底
        if not fallback_config:
            default_config = {
                "primary": {
                    "type": "text",
                    "split_method": "character",
                    "chunk_size": self.max_chunk_size,
                    "chunk_overlap": self.chunk_overlap,
                    "include_metadata": self.include_metadata,
                }
            }
            return default_config

        # 用户提供了配置，完全以用户配置为准，不做任何补充
        return fallback_config

    def _create_fallback_splitter(self, level: str) -> BaseSplitter:
        """根据配置创建退化分割器"""
        config = self.fallback_config[level].copy()  # 复制配置避免修改原配置
        splitter_type = config.pop("type")  # 取出type并从config中移除

        # 提前处理需要转换为enum的参数
        if "split_method" in config and isinstance(config["split_method"], str):
            config["split_method"] = SplitMethod.from_str(config["split_method"])

        if splitter_type == "semantic":
            return SemanticSplitter(**config)  # 直接展开剩余所有参数
        elif splitter_type == "text":
            return TextSplitter(**config)  # 直接展开剩余所有参数
        else:
            raise ValueError(f"不支持的退化分割器类型: {splitter_type}")

    def _handle_oversized_nodes(
        self, oversized_nodes: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """处理超大节点"""
        result = []

        for node in oversized_nodes:
            # 第一级退化
            primary_splitter = self._create_fallback_splitter("primary")

            primary_chunks = primary_splitter.split_text(
                node["content"], node["metadata"]
            )

            # 检查一级退化后是否还有超大节点
            still_oversized = []
            for chunk in primary_chunks:
                if len(chunk["content"]) > self.max_chunk_size:
                    still_oversized.append(chunk)
                else:
                    chunk["metadata"][
                        "split_method"
                    ] = f"hierarchical+{self.fallback_config['primary']['type']}"
                    result.append(chunk)

            # 第二级退化（最终兜底）
            if still_oversized:
                # 添加安全检查
                if "secondary" in self.fallback_config:
                    secondary_splitter = self._create_fallback_splitter("secondary")

                    for chunk in still_oversized:
                        final_chunks = secondary_splitter.split_text(
                            chunk["content"], chunk["metadata"]
                        )
                        for final_chunk in final_chunks:
                            final_chunk["metadata"]["split_method"] = (
                                f"hierarchical+{self.fallback_config['primary']['type']}"
                                f"+{self.fallback_config['secondary']['type']}"
                            )
                        result.extend(final_chunks)
                else:
                    # 如果经过所有退化后仍有超大节点，强制截断
                    for chunk in still_oversized:
                        content = chunk["content"]
                        metadata = chunk["metadata"]
                        
                        # 强制按最大长度截断
                        max_safe_length = min(300, self.max_chunk_size // 2)
                        
                        while len(content) > max_safe_length:
                            # 截断并创建新chunk
                            truncated_content = content[:max_safe_length]
                            truncated_metadata = metadata.copy()
                            truncated_metadata["split_method"] = "hierarchical+emergency_truncation"
                            truncated_metadata["truncated"] = True
                            
                            result.append({
                                "content": truncated_content,
                                "metadata": truncated_metadata
                            })
                            
                            content = content[max_safe_length:]
                        
                        # 处理剩余部分
                        if content:
                            final_metadata = metadata.copy()
                            final_metadata["split_method"] = "hierarchical+emergency_truncation"
                            final_metadata["truncated"] = True
                            
                            result.append({
                                "content": content,
                                "metadata": final_metadata
                            })

        return result

    # 修改 split 方法
    def split(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """将文档列表分割成更小的块"""
        if not documents:
            return []

        result = []

        for doc in documents:
            content = doc.get("content", "")
            metadata = doc.get("metadata", {})
            if not content:
                continue

            chunks = self.split_text(content, metadata)
            result.extend(chunks)

        return result

    def split_text(
        self, text: str, metadata: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """将文本分割成节点对象"""
        if not text:
            return []

        metadata = metadata or {}
        # 创建Document对象
        doc = Document(text=text, metadata=metadata)

        # 使用parser分割文档
        try:
            nodes = self.parser.get_nodes_from_documents([doc])
        except Exception as e:
            # 如果分割过程中出现异常，直接抛出，不再退化
            raise ValueError(f"HierarchicalNodeParser分割失败: {e}")

        # 转换为字典格式并检查是否有超过最大chunk_size的节点
        result = []
        oversized_nodes = []

        for i, node in enumerate(nodes):
            node_metadata = metadata.copy() if self.include_metadata else {}
            node_metadata.update(
                {
                    "chunk_index": i,
                    "total_chunks": len(nodes),
                    "chunk_size": self.chunk_sizes,
                    "split_method": "hierarchical",
                }
            )

            # 添加节点的元数据
            if hasattr(node, "metadata") and node.metadata:
                node_metadata.update(node.metadata)

            # 检查节点文本长度是否超过最大chunk_size
            if len(node.text) > self.max_chunk_size:
                oversized_nodes.append(
                    {"content": node.text, "metadata": node_metadata}
                )
            else:
                result.append({"content": node.text, "metadata": node_metadata})

        # 如果有超过最大chunk_size的节点，使用配置化的退化处理
        if oversized_nodes:

            fallback_chunks = self._handle_oversized_nodes(oversized_nodes)
            result.extend(fallback_chunks)

        return result


class SentenceSplitter(MetadataAwareTextSplitter, BaseSplitter):
    """Parse text with a preference for complete sentences.

    In general, this class tries to keep sentences and paragraphs together. Therefore
    compared to the original TokenTextSplitter, there are less likely to be
    hanging sentences or parts of sentences at the end of the node chunk.
    """

    chunk_size: int = Field(
        default=DEFAULT_CHUNK_SIZE,
        description="The token chunk size for each chunk.",
        gt=0,
    )
    chunk_overlap: int = Field(
        default=SENTENCE_CHUNK_OVERLAP,
        description="The token overlap of each chunk when splitting.",
        gte=0,
    )
    separator: str = Field(
        default=" ", description="Default separator for splitting into words"
    )
    paragraph_separator: str = Field(
        default=DEFAULT_PARAGRAPH_SEP, description="Separator between paragraphs."
    )
    secondary_chunking_regex: str = Field(
        default=CHUNKING_REGEX, description="Backup regex for splitting into sentences."
    )

    _chunking_tokenizer_fn: Callable[[str], List[str]] = PrivateAttr()
    _tokenizer: Callable = PrivateAttr()
    _split_fns: List[Callable] = PrivateAttr()
    _sub_sentence_split_fns: List[Callable] = PrivateAttr()

    def __init__(
        self,
        separator: str = " ",
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = SENTENCE_CHUNK_OVERLAP,
        tokenizer: Optional[Callable] = None,
        paragraph_separator: str = DEFAULT_PARAGRAPH_SEP,
        chunking_tokenizer_fn: Optional[Callable[[str], List[str]]] = None,
        secondary_chunking_regex: str = CHUNKING_REGEX,
        callback_manager: Optional[CallbackManager] = None,
        include_metadata: bool = True,
        include_prev_next_rel: bool = True,
        id_func: Optional[Callable[[int, Document], str]] = None,
    ):
        """Initialize with parameters."""
        if chunk_overlap > chunk_size:
            raise ValueError(
                f"Got a larger chunk overlap ({chunk_overlap}) than chunk size "
                f"({chunk_size}), should be smaller."
            )
        id_func = id_func or default_id_func
        callback_manager = callback_manager or CallbackManager([])

        # 先调用父类初始化
        super().__init__(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            secondary_chunking_regex=secondary_chunking_regex,
            separator=separator,
            paragraph_separator=paragraph_separator,
            callback_manager=callback_manager,
            include_metadata=include_metadata,
            include_prev_next_rel=include_prev_next_rel,
            id_func=id_func,
        )

        # 然后初始化私有属性
        self._init_private_attrs(
            tokenizer,
            chunking_tokenizer_fn,
            paragraph_separator,
            secondary_chunking_regex,
            separator,
        )

    def _init_private_attrs(
        self,
        tokenizer,
        chunking_tokenizer_fn,
        paragraph_separator,
        secondary_chunking_regex,
        separator,
    ):
        """初始化私有属性"""
        self._chunking_tokenizer_fn = (
            chunking_tokenizer_fn or split_by_sentence_tokenizer()
        )
        self._tokenizer = tokenizer or get_tokenizer()
        # print(self._tokenizer)  # 注释掉这行

        self._split_fns = [
            split_by_sep(paragraph_separator),
            self._chunking_tokenizer_fn,
        ]

        self._sub_sentence_split_fns = [
            split_by_regex(secondary_chunking_regex),
            split_by_sep(separator),
            split_by_char(),
        ]

    @classmethod
    def from_defaults(
        cls,
        separator: str = " ",
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = SENTENCE_CHUNK_OVERLAP,
        tokenizer: Optional[Callable] = None,
        paragraph_separator: str = DEFAULT_PARAGRAPH_SEP,
        chunking_tokenizer_fn: Optional[Callable[[str], List[str]]] = None,
        secondary_chunking_regex: str = CHUNKING_REGEX,
        callback_manager: Optional[CallbackManager] = None,
        include_metadata: bool = True,
        include_prev_next_rel: bool = True,
    ) -> "SentenceSplitter":
        """Initialize with parameters."""
        callback_manager = callback_manager or CallbackManager([])
        return cls(
            separator=separator,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            tokenizer=tokenizer,
            paragraph_separator=paragraph_separator,
            chunking_tokenizer_fn=chunking_tokenizer_fn,
            secondary_chunking_regex=secondary_chunking_regex,
            callback_manager=callback_manager,
            include_metadata=include_metadata,
            include_prev_next_rel=include_prev_next_rel,
        )

    @classmethod
    def class_name(cls) -> str:
        return "SentenceSplitter"

    def split_text_metadata_aware(self, text: str, metadata_str: str) -> List[str]:
        metadata_len = len(self._tokenizer(metadata_str))
        effective_chunk_size = self.chunk_size
        if effective_chunk_size <= 0:
            raise ValueError(
                f"Metadata length ({metadata_len}) is longer than chunk size "
                f"({self.chunk_size}). Consider increasing the chunk size or "
                "decreasing the size of your metadata to avoid this."
            )
        elif effective_chunk_size < 50:
            print(
                f"Metadata length ({metadata_len}) is close to chunk size "
                f"({self.chunk_size}). Resulting chunks are less than 50 tokens. "
                "Consider increasing the chunk size or decreasing the size of "
                "your metadata to avoid this.",
                flush=True,
            )

        return self._split_text(text, chunk_size=effective_chunk_size)

    def split_text(self, text: str) -> List[str]:
        return self._split_text(text, chunk_size=self.chunk_size)

    def _split_text(self, text: str, chunk_size: int) -> List[str]:
        """
        _Split incoming text and return chunks with overlap size.

        Has a preference for complete sentences, phrases, and minimal overlap.
        """
        if text == "":
            return [text]

        with self.callback_manager.event(
            CBEventType.CHUNKING, payload={EventPayload.CHUNKS: [text]}
        ) as event:
            splits = self._split(text, chunk_size)
            chunks = self._merge(splits, chunk_size)

            event.on_end(payload={EventPayload.CHUNKS: chunks})

        return chunks

    def _split(self, text: str, chunk_size: int) -> List[_Split]:
        r"""Break text into splits that are smaller than chunk size.

        The order of splitting is:
        1. split by paragraph separator
        2. split by chunking tokenizer (default is nltk sentence tokenizer)
        3. split by second chunking regex (default is "[^,\.;]+[,\.;]?")
        4. split by default separator (" ")

        """
        token_size = self._token_size(text)
        if token_size <= chunk_size:
            return [_Split(text, is_sentence=True, token_size=token_size)]

        text_splits_by_fns, is_sentence = self._get_splits_by_fns(text)

        text_splits = []
        for text_split_by_fns in text_splits_by_fns:
            token_size = self._token_size(text_split_by_fns)
            if token_size <= chunk_size:
                text_splits.append(
                    _Split(
                        text_split_by_fns,
                        is_sentence=is_sentence,
                        token_size=token_size,
                    )
                )
            else:
                recursive_text_splits = self._split(
                    text_split_by_fns, chunk_size=chunk_size
                )
                text_splits.extend(recursive_text_splits)
        return text_splits

    def _merge(self, splits: List[_Split], chunk_size: int) -> List[str]:
        """Merge splits into chunks."""
        chunks: List[str] = []
        cur_chunk: List[Tuple[str, int]] = []  # list of (text, length)
        last_chunk: List[Tuple[str, int]] = []
        cur_chunk_len = 0
        new_chunk = True

        def close_chunk() -> None:
            nonlocal chunks, cur_chunk, last_chunk, cur_chunk_len, new_chunk

            chunks.append("".join([text for text, length in cur_chunk]))
            last_chunk = cur_chunk
            cur_chunk = []
            cur_chunk_len = 0
            new_chunk = True

            # add overlap to the next chunk using the last one first
            # there is a small issue with this logic. If the chunk directly after
            # the overlap is really big, then we could go over the chunk_size, and
            # in theory the correct thing to do would be to remove some/all of the
            # overlap. However, it would complicate the logic further without
            # much real world benefit, so it's not implemented now.
            if len(last_chunk) > 0:
                last_index = len(last_chunk) - 1
                while (
                    last_index >= 0
                    and cur_chunk_len + last_chunk[last_index][1] <= self.chunk_overlap
                ):
                    text, length = last_chunk[last_index]
                    cur_chunk_len += length
                    cur_chunk.insert(0, (text, length))
                    last_index -= 1

        while len(splits) > 0:
            cur_split = splits[0]
            if cur_split.token_size > chunk_size:
                raise ValueError("Single token exceeded chunk size")
            if cur_chunk_len + cur_split.token_size > chunk_size and not new_chunk:
                # if adding split to current chunk exceeds chunk size: close out chunk
                close_chunk()
            else:
                if (
                    cur_split.is_sentence
                    or cur_chunk_len + cur_split.token_size <= chunk_size
                    or new_chunk  # new chunk, always add at least one split
                ):
                    # add split to chunk
                    cur_chunk_len += cur_split.token_size
                    cur_chunk.append((cur_split.text, cur_split.token_size))
                    splits.pop(0)
                    new_chunk = False
                else:
                    # close out chunk
                    close_chunk()

        # handle the last chunk
        if not new_chunk:
            chunk = "".join([text for text, length in cur_chunk])
            chunks.append(chunk)

        # run postprocessing to remove blank spaces
        return self._postprocess_chunks(chunks)

    def _postprocess_chunks(self, chunks: List[str]) -> List[str]:
        """Post-process chunks.
        Remove whitespace only chunks and remove leading and trailing whitespace.
        """
        new_chunks = []
        for chunk in chunks:
            stripped_chunk = chunk.strip()
            if stripped_chunk == "":
                continue
            new_chunks.append(stripped_chunk)
        return new_chunks

    def _token_size(self, text: str) -> int:
        return len(self._tokenizer(text))

    def _get_splits_by_fns(self, text: str) -> Tuple[List[str], bool]:
        for split_fn in self._split_fns:
            splits = split_fn(text)
            if len(splits) > 1:
                return splits, True

        for split_fn in self._sub_sentence_split_fns:
            splits = split_fn(text)
            if len(splits) > 1:
                break

        return splits, False

    def split(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """将文档列表分割成更小的块"""
        if not documents:
            return []

        result = []
        # 移除进度条，直接遍历文档
        for doc in documents:
            content = doc.get("content", "")
            metadata = doc.get("metadata", {})
            if not content:
                continue

            chunks = self.split_text(content, metadata)
            result.extend(chunks)

        return result


def _add_parent_child_relationship(parent_node: BaseNode, child_node: BaseNode) -> None:
    """Add parent/child relationship between nodes."""
    child_list = parent_node.relationships.get(NodeRelationship.CHILD, [])
    child_list.append(child_node.as_related_node_info())
    parent_node.relationships[NodeRelationship.CHILD] = child_list

    child_node.relationships[NodeRelationship.PARENT] = (
        parent_node.as_related_node_info()
    )


def get_leaf_nodes(nodes: List[BaseNode]) -> List[BaseNode]:
    """Get leaf nodes."""
    leaf_nodes = []
    for node in nodes:
        if NodeRelationship.CHILD not in node.relationships:
            leaf_nodes.append(node)
    return leaf_nodes


def get_root_nodes(nodes: List[BaseNode]) -> List[BaseNode]:
    """Get root nodes."""
    root_nodes = []
    for node in nodes:
        if NodeRelationship.PARENT not in node.relationships:
            root_nodes.append(node)
    return root_nodes


def get_child_nodes(nodes: List[BaseNode], all_nodes: List[BaseNode]) -> List[BaseNode]:
    """Get child nodes of nodes from given all_nodes."""
    children_ids = []
    for node in nodes:
        if NodeRelationship.CHILD not in node.relationships:
            continue

        children_ids.extend(
            [r.node_id for r in node.relationships[NodeRelationship.CHILD]]
        )

    child_nodes = []
    for candidate_node in all_nodes:
        if candidate_node.node_id not in children_ids:
            continue
        child_nodes.append(candidate_node)

    return child_nodes


def get_deeper_nodes(nodes: List[BaseNode], depth: int = 1) -> List[BaseNode]:
    """Get children of root nodes in given nodes that have given depth."""
    if depth < 0:
        raise ValueError("Depth cannot be a negative number!")
    root_nodes = get_root_nodes(nodes)
    if not root_nodes:
        raise ValueError("There is no root nodes in given nodes!")

    deeper_nodes = root_nodes
    for _ in range(depth):
        deeper_nodes = get_child_nodes(deeper_nodes, nodes)

    return deeper_nodes


class HierarchicalNodeParser(NodeParser, BaseSplitter):
    """Hierarchical node parser.

    Splits a document into a recursive hierarchy Nodes using a NodeParser.

    NOTE: this will return a hierarchy of nodes in a flat list, where there will be
    overlap between parent nodes (e.g. with a bigger chunk size), and child nodes
    per parent (e.g. with a smaller chunk size).

    For instance, this may return a list of nodes like:
    - list of top-level nodes with chunk size 2048
    - list of second-level nodes, where each node is a child of a top-level node,
      chunk size 512
    - list of third-level nodes, where each node is a child of a second-level node,
      chunk size 128
    """

    chunk_sizes: Optional[List[int]] = Field(
        default=None,
        description=(
            "The chunk sizes to use when splitting documents, in order of level."
        ),
    )
    node_parser_ids: List[str] = Field(
        default_factory=list,
        description=(
            "List of ids for the node parsers to use when splitting documents, "
            + "in order of level (first id used for first level, etc.)."
        ),
    )
    node_parser_map: Dict[str, NodeParser] = Field(
        description="Map of node parser id to node parser.",
    )

    @classmethod
    def from_defaults(
        cls,
        chunk_sizes: Optional[List[int]] = None,
        chunk_overlap: int = 20,
        node_parser_ids: Optional[List[str]] = None,
        node_parser_map: Optional[Dict[str, NodeParser]] = None,
        include_metadata: bool = True,
        include_prev_next_rel: bool = True,
        callback_manager: Optional[CallbackManager] = None,
    ) -> "HierarchicalNodeParser":
        callback_manager = callback_manager or CallbackManager([])

        if node_parser_ids is None:
            if chunk_sizes is None:
                chunk_sizes = [2048, 512, 128]

            node_parser_ids = [f"chunk_size_{chunk_size}" for chunk_size in chunk_sizes]
            node_parser_map = {}
            for chunk_size, node_parser_id in zip(chunk_sizes, node_parser_ids):
                node_parser_map[node_parser_id] = SentenceSplitter.from_defaults(
                    chunk_size=chunk_size,
                    callback_manager=callback_manager,
                    chunk_overlap=chunk_overlap,
                    include_metadata=include_metadata,
                    include_prev_next_rel=include_prev_next_rel,
                )
        else:
            if chunk_sizes is not None:
                raise ValueError("Cannot specify both node_parser_ids and chunk_sizes.")
            if node_parser_map is None:
                raise ValueError(
                    "Must specify node_parser_map if using node_parser_ids."
                )

        return cls(
            chunk_sizes=chunk_sizes,
            node_parser_ids=node_parser_ids,
            node_parser_map=node_parser_map,
            include_metadata=include_metadata,
            include_prev_next_rel=include_prev_next_rel,
            callback_manager=callback_manager,
        )

    @classmethod
    def class_name(cls) -> str:
        return "HierarchicalNodeParser"

    def _recursively_get_nodes_from_nodes(
        self,
        nodes: List[BaseNode],
        level: int,
        show_progress: bool = False,
    ) -> List[BaseNode]:
        """Recursively get nodes from nodes."""
        if level >= len(self.node_parser_ids):
            raise ValueError(
                f"Level {level} is greater than number of text "
                f"splitters ({len(self.node_parser_ids)})."
            )

        sub_nodes = []
        for node in nodes:
            cur_sub_nodes = self.node_parser_map[
                self.node_parser_ids[level]
            ].get_nodes_from_documents([node])
            # add parent relationship from sub node to parent node
            # add child relationship from parent node to sub node
            # relationships for the top-level document objects that            # NOTE: Only add relationships if level > 0, since we don't want to add we are splitting
            if level > 0:
                for sub_node in cur_sub_nodes:
                    _add_parent_child_relationship(
                        parent_node=node,
                        child_node=sub_node,
                    )

            sub_nodes.extend(cur_sub_nodes)

        # now for each sub-node, recursively split into sub-sub-nodes, and add
        if level < len(self.node_parser_ids) - 1:
            sub_sub_nodes = self._recursively_get_nodes_from_nodes(
                sub_nodes,
                level + 1,
                show_progress=show_progress,
            )
        else:
            sub_sub_nodes = []

        return sub_nodes + sub_sub_nodes

    def get_nodes_from_documents(
        self,
        documents: Sequence[Document],
        show_progress: bool = False,
        **kwargs: Any,
    ) -> List[BaseNode]:
        """Parse document into nodes.

        Args:
            documents (Sequence[Document]): documents to parse
            include_metadata (bool): whether to include metadata in nodes

        """
        with self.callback_manager.event(
            CBEventType.NODE_PARSING, payload={EventPayload.DOCUMENTS: documents}
        ) as event:
            all_nodes: List[BaseNode] = []

            for doc in documents:
                nodes_from_doc = self._recursively_get_nodes_from_nodes([doc], 0)
                all_nodes.extend(nodes_from_doc)

            event.on_end(payload={EventPayload.NODES: all_nodes})

        return all_nodes

    # Unused abstract method
    def _parse_nodes(
        self, nodes: Sequence[BaseNode], show_progress: bool = False, **kwargs: Any
    ) -> List[BaseNode]:
        return list(nodes)

    def split(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """分割文档列表"""
        result = []
        # 移除进度条，直接遍历文档
        for doc in documents:
            chunks = self.split_text(doc["content"], doc.get("metadata", {}))
            result.extend(chunks)
        return result

    def split_text(
        self, text: str, metadata: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """分割单个文本"""
        if metadata is None:
            metadata = {}

        # 创建Document对象
        from llama_index.core.schema import Document

        document = Document(text=text, metadata=metadata)

        # 使用HierarchicalNodeParser的get_nodes_from_documents方法
        nodes = self.get_nodes_from_documents([document])

        # 转换为字典格式
        chunks = []
        for node in nodes:
            chunk_data = {
                "content": node.text,
                "metadata": {
                    **metadata,
                    **node.metadata,
                    "node_id": node.node_id,
                },
            }

            # 提取标题信息
            if node.text.strip().startswith("#"):
                # 这是一个标题节点
                chunk_data["metadata"]["node_type"] = "header"
                chunk_data["metadata"]["header_text"] = node.text.strip()
            else:
                # 这是一个正文节点
                chunk_data["metadata"]["node_type"] = "content"
                # 从header_path中提取所属标题
                if "header_path" in node.metadata:
                    chunk_data["metadata"]["parent_headers"] = node.metadata[
                        "header_path"
                    ]

            chunks.append(chunk_data)

        return chunks
