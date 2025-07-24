from typing import Any, Dict

from rag.pipeline.registry import ComponentRegistry
from utils.prompt import quick_fill

from .base_query import BaseQueryComponent


@ComponentRegistry.register("query", "abstraction")
class AbstractionComponent(BaseQueryComponent):
    """查询抽象组件 - 将具体查询抽象为更通用的问题"""

    def __init__(self, name: str, config: Dict[str, Any]):
        super().__init__(name, config)
        self.abstraction_level = config.get(
            "abstraction_level", "medium"
        )  # low, medium, high
        self.preserve_domain = config.get("preserve_domain", True)

    def _optimize_query(self, query: str) -> Dict[str, Any]:
        """执行查询抽象"""
        try:
            # 检测是否需要抽象
            if not self._needs_abstraction(query):
                return {
                    "abstracted_query": query,
                    "abstraction_needed": False,
                    "abstraction_level": "none",
                    "original_query": query,
                }

            # 执行抽象
            prompt = quick_fill("abstract", query=query)
            response = self._call_llm_with_retry(prompt)
            abstracted_query = self._extract_output(response)

            # 根据抽象级别调整结果
            abstracted_query = self._adjust_abstraction_level(query, abstracted_query)

            # 验证抽象结果
            if not self._validate_abstraction(query, abstracted_query):
                abstracted_query = query
                abstraction_needed = False
            else:
                abstraction_needed = True

            return {
                "abstracted_query": abstracted_query,
                "abstraction_needed": abstraction_needed,
                "abstraction_level": self.abstraction_level,
                "original_query": query,
                "domain_preserved": self.preserve_domain,
            }

        except Exception as e:
            self.logger.error(f"查询抽象失败: {e}")
            return {"abstracted_query": query, "abstraction_needed": False}

    def _needs_abstraction(self, query: str) -> bool:
        """判断查询是否需要抽象"""
        # 检查具体性指标
        specific_indicators = [
            # 具体的数字、日期、名称
            r"\d{4}",  # 年份
            r"\d+月",  # 月份
            r"\d+日",  # 日期
            r"\d+年",  # 年份
            r"[A-Z][a-z]+\s[A-Z][a-z]+",  # 人名
            r"[A-Z]{2,}",  # 缩写
        ]

        import re

        specific_count = sum(
            1 for pattern in specific_indicators if re.search(pattern, query)
        )

        # 检查专有名词
        proper_nouns = [
            "公司",
            "大学",
            "学院",
            "医院",
            "银行",
            "Company",
            "University",
            "Hospital",
        ]
        proper_noun_count = sum(1 for noun in proper_nouns if noun in query)

        # 如果包含具体信息，可能需要抽象
        return specific_count > 0 or proper_noun_count > 0 or len(query.split()) > 10

    def _adjust_abstraction_level(self, original: str, abstracted: str) -> str:
        """根据设定的抽象级别调整结果"""
        if self.abstraction_level == "low":
            # 低级抽象：保留更多细节
            words_original = original.split()
            words_abstracted = abstracted.split()

            # 如果抽象程度太高，混合一些原始词汇
            if len(words_abstracted) < len(words_original) * 0.7:
                # 保留一些关键词
                key_words = [w for w in words_original if len(w) > 3]
                if key_words:
                    abstracted += f" (涉及: {', '.join(key_words[:2])})"

        elif self.abstraction_level == "high":
            # 高级抽象：更加通用
            # 进一步简化表达
            abstracted = self._further_abstract(abstracted)

        return abstracted

    def _further_abstract(self, query: str) -> str:
        """进一步抽象查询"""
        # 替换具体词汇为通用词汇
        replacements = {
            "购买": "获取",
            "销售": "提供",
            "学习": "了解",
            "研究": "分析",
            "开发": "创建",
            "设计": "规划",
        }

        for specific, general in replacements.items():
            query = query.replace(specific, general)

        return query

    def _validate_abstraction(self, original: str, abstracted: str) -> bool:
        """验证抽象结果的质量"""
        # 基本验证
        if not abstracted or len(abstracted) < 5:
            return False

        # 检查是否过度抽象
        if len(abstracted.split()) < 3:
            return False

        # 检查是否保留了核心意图
        original_words = set(original.lower().split())
        abstracted_words = set(abstracted.lower().split())

        # 至少应该有一些词汇重叠或语义相关
        overlap = len(original_words & abstracted_words)

        return overlap > 0 or len(abstracted_words) >= len(original_words) * 0.3
