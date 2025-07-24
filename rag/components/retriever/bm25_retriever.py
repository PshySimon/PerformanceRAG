import math
import os
import pickle
import re
from collections import Counter
from typing import Any, Dict, List, Optional

from .base_retriever import BaseRetrieverComponent

try:
    import jieba
except ImportError:
    jieba = None


class BM25RetrieverComponent(BaseRetrieverComponent):
    """BM25检索器组件"""

    def __init__(self, name: str, config: Dict[str, Any]):
        super().__init__(name, config)

        if jieba is None:
            raise ImportError("请安装jieba包: pip install jieba")

        # BM25参数
        self.k1 = config.get("k1", 1.5)
        self.b = config.get("b", 0.75)

        # 索引配置
        self.storage_path = config.get("storage_path", "./data/bm25_index")
        self.auto_load = config.get("auto_load", True)

        # 内部状态
        self.index_data = None
        self.doc_store = None
        self.doc_lengths = None
        self.avg_doc_length = 0
        self.N = 0

    def _do_initialize(self):
        """初始化BM25检索器"""
        try:
            if self.auto_load:
                self._connect_to_index()

            if self.debug:
                self.logger.debug(f"BM25检索器初始化完成，文档数量: {self.N}")

        except Exception as e:
            self.logger.error(f"初始化BM25检索器失败: {e}")
            raise

    def _connect_to_index(self):
        """连接到BM25索引"""
        try:
            index_path = self._get_index_path()

            if not os.path.exists(index_path):
                if self.debug:
                    self.logger.debug("BM25索引文件不存在")
                return

            with open(index_path, "rb") as f:
                index_data = pickle.load(f)

            self.index_data = index_data.get("index_data", {})
            self.doc_store = index_data.get("doc_store", {})
            self.doc_lengths = index_data.get("doc_lengths", {})
            self.avg_doc_length = index_data.get("avg_doc_length", 0)
            self.N = index_data.get("N", 0)

            if self.debug:
                self.logger.debug(f"成功加载BM25索引，文档数量: {self.N}")

        except Exception as e:
            self.logger.error(f"加载BM25索引失败: {e}")
            # 重置状态
            self.index_data = {}
            self.doc_store = {}
            self.doc_lengths = {}
            self.avg_doc_length = 0
            self.N = 0

    def _get_index_path(self) -> str:
        """获取索引文件路径"""
        return os.path.join(self.storage_path, f"{self.index_name}.pkl")

    def _tokenize(self, text: str) -> List[str]:
        """文本分词"""
        # 使用jieba分词
        tokens = list(jieba.cut(text, cut_all=False))
        final_tokens = []

        for token in tokens:
            # 对每个jieba分出来的词再用正则分割
            final_tokens.extend([t for t in re.split(r"\W+", token) if t])

        return [token for token in final_tokens if len(token) > 1]

    def retrieve(
        self, query: str, top_k: Optional[int] = None, **kwargs
    ) -> List[Dict[str, Any]]:
        """检索文档"""
        use_top_k = top_k or self.top_k

        try:
            if not self.index_data or not self.doc_store:
                if self.debug:
                    self.logger.debug("BM25索引为空，返回空结果")
                return []

            # 查询分词
            query_tokens = self._tokenize(query)
            if self.debug:
                self.logger.debug(f"查询: '{query}'")
                self.logger.debug(f"分词结果: {query_tokens}")

            if not query_tokens:
                if self.debug:
                    self.logger.debug("分词结果为空，返回空结果")
                return []

            # 计算BM25分数
            scores = self._calculate_bm25_scores(query_tokens)

            # 获取top-k结果
            top_docs = scores.most_common(use_top_k)

            if self.debug:
                self.logger.debug(f"所有文档分数统计: 共 {len(scores)} 个文档有分数")
                if len(scores) > 0:
                    max_score = max(scores.values())
                    min_score = min(scores.values())
                    avg_score = sum(scores.values()) / len(scores)
                    self.logger.debug(
                        f"分数范围: 最高={max_score:.4f}, 最低={min_score:.4f}, 平均={avg_score:.4f}"
                    )

            # 格式化结果
            results = []
            for i, (doc_id, score) in enumerate(top_docs):
                if score > self.similarity_threshold:
                    doc_data = self.doc_store.get(doc_id)
                    if doc_data:
                        result = {
                            "id": doc_id,
                            "score": float(score),
                            "content": doc_data.get("content", ""),
                            "metadata": doc_data.get("metadata", {}),
                        }
                        results.append(result)

                        if self.debug:
                            content_preview = doc_data.get("content", "")[:100].replace(
                                "\n", " "
                            )
                            title = doc_data.get("metadata", {}).get("title", "无标题")
                            self.logger.debug(
                                f"Top-{i+1}: 文档ID={doc_id}, 分数={score:.4f}, 标题='{title}', 内容预览='{content_preview}...'"
                            )
                    else:
                        if self.debug:
                            self.logger.debug(
                                f"文档ID={doc_id}的分数{score:.4f}低于阈值{self.similarity_threshold}，已过滤"
                            )

            if self.debug:
                self.logger.debug(
                    f"BM25检索完成: 候选文档{len(scores)}个 -> Top-{use_top_k}筛选 -> 阈值过滤 -> 最终返回{len(results)}个结果"
                )

            return results

        except Exception as e:
            self.logger.error(f"BM25检索失败: {e}")
            return []

    def _calculate_bm25_scores(self, query_tokens: List[str]) -> Counter:
        """计算BM25分数"""
        scores = Counter()

        if self.debug:
            self.logger.debug(f"开始计算BM25分数，查询词数量: {len(query_tokens)}")
            self.logger.debug(
                f"索引统计: 总文档数N={self.N}, 平均文档长度={self.avg_doc_length:.2f}"
            )
            self.logger.debug(f"BM25参数: k1={self.k1}, b={self.b}")

        for i, term in enumerate(query_tokens):
            doc_ids = self.index_data.get(term, [])

            if self.debug:
                self.logger.debug(
                    f"\n--- 处理查询词 {i+1}/{len(query_tokens)}: '{term}' ---"
                )
                self.logger.debug(f"匹配文档数: {len(doc_ids)}")

            if not doc_ids:
                if self.debug:
                    self.logger.debug(f"词 '{term}' 在索引中未找到匹配文档")
                continue

            # 计算IDF
            df = len(doc_ids)
            idf = math.log((self.N - df + 0.5) / (df + 0.5) + 1)

            if self.debug:
                self.logger.debug(f"IDF计算: df={df}, idf={idf:.4f}")
                self.logger.debug(
                    f"匹配的文档ID: {doc_ids[:10]}{'...' if len(doc_ids) > 10 else ''}"
                )

            for j, doc_id in enumerate(doc_ids):
                # 计算TF
                doc_data = self.doc_store.get(doc_id, {})
                doc_content = doc_data.get("content", "")
                tf = doc_content.split().count(term)

                # 文档长度
                doc_len = self.doc_lengths.get(doc_id, 0)

                # BM25分数
                score = (
                    idf
                    * (tf * (self.k1 + 1))
                    / (
                        tf
                        + self.k1
                        * (1 - self.b + self.b * doc_len / self.avg_doc_length)
                    )
                )
                scores[doc_id] += score

                # 详细日志（只显示前几个文档的详细计算过程）
                if self.debug and j < 3:
                    title = doc_data.get("metadata", {}).get("title", "无标题")
                    self.logger.debug(
                        f"  文档 '{title}' (ID={doc_id}): tf={tf}, doc_len={doc_len}, 本词贡献分数={score:.4f}, 累计分数={scores[doc_id]:.4f}"
                    )
                elif self.debug and j == 3 and len(doc_ids) > 3:
                    self.logger.debug(
                        f"  ... 还有 {len(doc_ids) - 3} 个文档的计算过程已省略"
                    )

        # 修复：将return语句移到for循环外部
        if self.debug:
            self.logger.debug("\n=== BM25分数计算完成 ===")
            top_5_scores = Counter(scores).most_common(5)
            self.logger.debug(
                f"Top-5分数预览: {[(doc_id, f'{score:.4f}') for doc_id, score in top_5_scores]}"
            )

        return scores
