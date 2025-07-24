from typing import Any, Dict

from rag.pipeline.registry import ComponentRegistry
from utils.prompt import quick_fill

from .base_query import BaseQueryComponent


@ComponentRegistry.register("query", "expansion")
class ExpansionComponent(BaseQueryComponent):
    """查询扩展组件 - 生成多个相关查询表达"""

    def __init__(self, name: str, config: Dict[str, Any]):
        super().__init__(name, config)
        self.expansion_type = config.get(
            "expansion_type", "multi_query"
        )  # multi_query, hyde, rewrite
        self.num_queries = config.get("num_queries", 3)

    def _optimize_query(self, query: str) -> Dict[str, Any]:
        """执行查询扩展"""
        if self.expansion_type == "multi_query":
            return self._multi_query_expansion(query)
        elif self.expansion_type == "hyde":
            return self._hyde_expansion(query)
        elif self.expansion_type == "rewrite":
            return self._rewrite_expansion(query)
        else:
            self.logger.warning(f"未知的扩展类型: {self.expansion_type}")
            return {"expanded_queries": [query]}

    def _multi_query_expansion(self, query: str) -> Dict[str, Any]:
        """多查询扩展"""
        try:
            prompt = quick_fill("multi_query", query=query, n=self.num_queries)
            response = self._call_llm_with_retry(prompt)
            content = self._extract_output(response)

            # 解析多个查询
            expanded_queries = [
                line.strip() for line in content.splitlines() if line.strip()
            ]

            # 确保包含原始查询
            if query not in expanded_queries:
                expanded_queries.insert(0, query)

            return {
                "expanded_queries": expanded_queries,
                "expansion_type": "multi_query",
                "original_query": query,
            }

        except Exception as e:
            self.logger.error(f"多查询扩展失败: {e}")
            return {"expanded_queries": [query]}

    def _hyde_expansion(self, query: str) -> Dict[str, Any]:
        """HyDE (Hypothetical Document Embeddings) 扩展"""
        try:
            prompt = quick_fill("hyde", query=query)
            response = self._call_llm_with_retry(prompt)
            hypothetical_answer = self._extract_output(response)

            return {
                "expanded_queries": [query, hypothetical_answer],
                "hypothetical_answer": hypothetical_answer,
                "expansion_type": "hyde",
                "original_query": query,
            }

        except Exception as e:
            self.logger.error(f"HyDE扩展失败: {e}")
            return {"expanded_queries": [query]}

    def _rewrite_expansion(self, query: str) -> Dict[str, Any]:
        """查询重写扩展"""
        try:
            prompt = quick_fill("rewrite", query=query)
            response = self._call_llm_with_retry(prompt)
            rewritten_query = self._extract_output(response)

            return {
                "expanded_queries": [query, rewritten_query],
                "rewritten_query": rewritten_query,
                "expansion_type": "rewrite",
                "original_query": query,
            }

        except Exception as e:
            self.logger.error(f"查询重写失败: {e}")
            return {"expanded_queries": [query]}
