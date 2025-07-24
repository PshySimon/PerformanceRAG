from abc import abstractmethod
from typing import Any, Dict, List

from rag.components.base import Component
from utils.logger import get_logger


class BaseRerankerComponent(Component):
    """重排组件基类"""

    def __init__(self, name: str, config: Dict[str, Any]):
        super().__init__(name, config)
        self.top_k = config.get("top_k", 10)
        self.score_threshold = config.get("score_threshold", 0.0)
        self.logger = get_logger(__name__)

    def _do_initialize(self):
        """初始化重排组件"""
        self.logger.info(f"重排组件 {self.name} 初始化成功")

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """处理重排数据"""
        query = data.get("query", "")
        documents = data.get("documents", [])

        if not query:
            self.logger.warning("输入数据中没有找到query字段")
            return data

        if not documents:
            self.logger.warning("输入数据中没有找到documents字段")
            return data

        try:
            # 执行重排
            reranked_docs = self._rerank(query, documents)

            # 更新数据
            result = data.copy()
            result["documents"] = reranked_docs
            result["rerank_metadata"] = {
                "component": self.name,
                "original_count": len(documents),
                "reranked_count": len(reranked_docs),
                "top_k": self.top_k,
            }

            if self.debug:
                self.logger.debug(f"重排完成: {len(documents)} -> {len(reranked_docs)}")

            return result

        except Exception as e:
            self.logger.error(f"重排失败: {e}")
            return data

    @abstractmethod
    def _rerank(
        self, query: str, documents: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """执行具体的重排逻辑"""
        pass

    def _filter_by_score(self, scored_docs: List[tuple]) -> List[Dict[str, Any]]:
        """根据分数阈值过滤文档"""
        filtered = [
            (doc, score) for doc, score in scored_docs if score >= self.score_threshold
        ]
        # 按分数降序排序并取top_k
        filtered.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, score in filtered[: self.top_k]]
