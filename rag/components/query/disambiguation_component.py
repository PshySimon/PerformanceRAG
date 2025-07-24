from typing import Any, Dict

from rag.pipeline.registry import ComponentRegistry
from utils.prompt import quick_fill

from .base_query import BaseQueryComponent


@ComponentRegistry.register("query", "disambiguation")
class DisambiguationComponent(BaseQueryComponent):
    """查询消歧组件 - 消除查询中的歧义"""

    def __init__(self, name: str, config: Dict[str, Any]):
        super().__init__(name, config)
        self.context_window = config.get("context_window", 3)  # 上下文窗口大小
        self.ambiguity_threshold = config.get("ambiguity_threshold", 0.7)

    def _optimize_query(self, query: str) -> Dict[str, Any]:
        """执行查询消歧"""
        try:
            # 检测是否存在歧义
            ambiguity_score = self._detect_ambiguity(query)

            if ambiguity_score < self.ambiguity_threshold:
                return {
                    "disambiguated_query": query,
                    "disambiguation_needed": False,
                    "ambiguity_score": ambiguity_score,
                    "original_query": query,
                }

            # 执行消歧
            prompt = quick_fill("disambiguate", query=query)
            response = self._call_llm_with_retry(prompt)
            disambiguated_query = self._extract_output(response)

            # 验证消歧结果
            if not disambiguated_query or len(disambiguated_query) < len(query) * 0.5:
                disambiguated_query = query
                disambiguation_needed = False
            else:
                disambiguation_needed = True

            return {
                "disambiguated_query": disambiguated_query,
                "disambiguation_needed": disambiguation_needed,
                "ambiguity_score": ambiguity_score,
                "original_query": query,
                "disambiguation_method": "llm_based",
            }

        except Exception as e:
            self.logger.error(f"查询消歧失败: {e}")
            return {"disambiguated_query": query, "disambiguation_needed": False}

    def _detect_ambiguity(self, query: str) -> float:
        """检测查询的歧义程度"""
        ambiguity_indicators = {
            # 代词
            "pronouns": [
                "它",
                "他",
                "她",
                "这",
                "那",
                "这个",
                "那个",
                "it",
                "this",
                "that",
                "they",
            ],
            # 模糊词汇
            "vague_terms": [
                "东西",
                "事情",
                "问题",
                "方面",
                "情况",
                "thing",
                "stuff",
                "issue",
                "aspect",
            ],
            # 多义词
            "polysemous": ["银行", "苹果", "橙子", "bank", "apple", "orange", "python"],
            # 缺少限定词
            "lack_qualifiers": True if len(query.split()) < 3 else False,
        }

        score = 0.0
        total_words = len(query.split())

        # 检查代词
        pronoun_count = sum(
            1 for word in ambiguity_indicators["pronouns"] if word in query.lower()
        )
        score += (pronoun_count / total_words) * 0.4

        # 检查模糊词汇
        vague_count = sum(
            1 for word in ambiguity_indicators["vague_terms"] if word in query.lower()
        )
        score += (vague_count / total_words) * 0.3

        # 检查多义词
        polysemous_count = sum(
            1 for word in ambiguity_indicators["polysemous"] if word in query.lower()
        )
        score += (polysemous_count / total_words) * 0.2

        # 检查查询长度（太短可能有歧义）
        if total_words < 3:
            score += 0.3
        elif total_words < 5:
            score += 0.1

        return min(score, 1.0)
