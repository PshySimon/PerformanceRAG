import re
from typing import Any, Dict, List

from utils.llm.factory import LLMFactory
from utils.prompt import PromptTemplate

from .base_reranker import BaseRerankerComponent


class LLMRerankerComponent(BaseRerankerComponent):
    """LLM重排组件"""

    def __init__(self, name: str, config: Dict[str, Any]):
        super().__init__(name, config)
        self.llm = None
        self.max_retries = config.get("max_retries", 3)
        self.temperature = config.get("temperature", 0.3)
        self.rerank_method = config.get(
            "method", "pairwise"
        )  # pairwise, listwise, pointwise

    def _do_initialize(self):
        """初始化LLM"""
        try:
            self.llm = LLMFactory.from_config()
            super()._do_initialize()
        except Exception as e:
            self.logger.error(f"LLM重排组件 {self.name} 初始化失败: {e}")
            raise

    def _rerank(
        self, query: str, documents: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """使用LLM进行重排"""
        if self.rerank_method == "listwise":
            return self._listwise_rerank(query, documents)
        elif self.rerank_method == "pointwise":
            return self._pointwise_rerank(query, documents)
        else:  # pairwise
            return self._pairwise_rerank(query, documents)

    def _listwise_rerank(
        self, query: str, documents: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """列表式重排：一次性对所有文档排序"""
        # 构建文档列表
        doc_list = []
        for i, doc in enumerate(documents):
            content = doc.get("content", doc.get("text", ""))
            doc_list.append(f"[{i+1}] {content[:200]}...")  # 截取前200字符

        # 构建提示
        prompt_template = PromptTemplate(
            "请对以下文档片段按照与问题的相关性进行排序：\n\n"
            "问题：{question}\n\n"
            "文档片段：\n{documents}\n\n"
            "请按相关性从高到低排序，只输出序号列表，格式如：[1,3,2,5,4]"
        )

        prompt = prompt_template.fill(
            {"question": query, "documents": "\n".join(doc_list)}
        )

        # 调用LLM
        response = self._call_llm_with_retry(prompt)

        # 解析排序结果
        try:
            # 提取序号列表
            match = re.search(r"\[(\d+(?:,\s*\d+)*)\]", response)
            if match:
                indices = [int(x.strip()) - 1 for x in match.group(1).split(",")]
                # 重新排序文档
                reranked = []
                for idx in indices:
                    if 0 <= idx < len(documents):
                        reranked.append(documents[idx])

                # 添加未排序的文档
                used_indices = set(indices)
                for i, doc in enumerate(documents):
                    if i not in used_indices:
                        reranked.append(doc)

                return reranked[: self.top_k]
        except Exception as e:
            self.logger.warning(f"解析LLM排序结果失败: {e}，返回原始顺序")

        return documents[: self.top_k]

    def _pointwise_rerank(
        self, query: str, documents: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """点式重排：为每个文档单独打分"""
        scored_docs = []

        for doc in documents:
            content = doc.get("content", doc.get("text", ""))

            prompt_template = PromptTemplate(
                "请评估以下文档与问题的相关性，给出0-10的分数：\n\n"
                "问题：{question}\n\n"
                "文档：{document}\n\n"
                "请只输出数字分数："
            )

            prompt = prompt_template.fill(
                {"question": query, "document": content[:500]}  # 截取前500字符
            )

            try:
                response = self._call_llm_with_retry(prompt)
                # 提取分数
                score_match = re.search(r"(\d+(?:\.\d+)?)", response)
                score = float(score_match.group(1)) if score_match else 0.0
                scored_docs.append((doc, score))
            except Exception as e:
                self.logger.warning(f"为文档打分失败: {e}")
                scored_docs.append((doc, 0.0))

        return self._filter_by_score(scored_docs)

    def _pairwise_rerank(
        self, query: str, documents: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """对式重排：两两比较文档相关性"""
        if len(documents) <= 1:
            return documents

        # 简化的冒泡排序式比较
        docs = documents.copy()
        n = len(docs)

        for i in range(n):
            for j in range(0, n - i - 1):
                if self._compare_documents(query, docs[j], docs[j + 1]):
                    docs[j], docs[j + 1] = docs[j + 1], docs[j]

        return docs[: self.top_k]

    def _compare_documents(
        self, query: str, doc1: Dict[str, Any], doc2: Dict[str, Any]
    ) -> bool:
        """比较两个文档的相关性，返回True表示doc2更相关"""
        content1 = doc1.get("content", doc1.get("text", ""))[:300]
        content2 = doc2.get("content", doc2.get("text", ""))[:300]

        prompt_template = PromptTemplate(
            "请比较以下两个文档哪个与问题更相关：\n\n"
            "问题：{question}\n\n"
            "文档A：{doc1}\n\n"
            "文档B：{doc2}\n\n"
            "请回答A或B："
        )

        prompt = prompt_template.fill(
            {"question": query, "doc1": content1, "doc2": content2}
        )

        try:
            response = self._call_llm_with_retry(prompt)
            return "B" in response.upper()
        except Exception as e:
            self.logger.warning(f"文档比较失败: {e}")
            return False

    def _call_llm_with_retry(self, prompt: str) -> str:
        """带重试的LLM调用"""
        for attempt in range(self.max_retries):
            try:
                response = self.llm.completion(prompt, temperature=self.temperature)
                return response
            except Exception as e:
                self.logger.warning(
                    f"LLM调用失败 (尝试 {attempt + 1}/{self.max_retries}): {e}"
                )
                if attempt == self.max_retries - 1:
                    raise
        return ""
