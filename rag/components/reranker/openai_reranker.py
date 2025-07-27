from typing import Any, Dict, List

import requests

from .base_reranker import BaseRerankerComponent


class OpenAIRerankerComponent(BaseRerankerComponent):
    """OpenAI重排组件"""

    def __init__(self, name: str, config: Dict[str, Any]):
        super().__init__(name, config)
        self.api_key = config.get("api_key")
        self.base_url = config.get("base_url", "https://api.openai.com/v1").rstrip("/")
        self.model = config.get("model", "gpt-3.5-turbo")
        self.timeout = config.get("timeout", 30)
        self.temperature = config.get("temperature", 0.1)
        self.max_tokens = config.get("max_tokens", 512)

        if not self.api_key:
            raise ValueError("OpenAI reranker需要配置api_key")

    def _do_initialize(self):
        """初始化OpenAI重排组件"""
        try:
            self.logger.info(f"OpenAI重排组件 {self.name} 初始化成功")
        except Exception as e:
            self.logger.error(f"OpenAI重排组件 {self.name} 初始化失败: {e}")
            raise

    def _rerank(
        self, query: str, documents: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """使用OpenAI API进行重排"""
        if not documents:
            return []

        try:
            # 检查是否是BGE reranker模型
            if "bge-reranker" in self.model.lower():
                return self._rerank_with_bge_api(query, documents)
            else:
                return self._rerank_with_chat_api(query, documents)

        except Exception as e:
            self.logger.error(f"OpenAI重排失败: {e}")
            return documents[: self.top_k]

    def _rerank_with_bge_api(
        self, query: str, documents: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """使用BGE reranker API进行重排"""
        # 准备文档内容
        doc_texts = []
        for doc in documents:
            content = doc.get("content", doc.get("text", ""))
            doc_texts.append(str(content))

        # 构建BGE reranker API请求
        url = f"{self.base_url}/rerank"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        data = {
            "model": self.model,
            "query": query,
            "documents": doc_texts,
            "top_k": self.top_k,
        }

        response = requests.post(url, headers=headers, json=data, timeout=self.timeout)

        if response.status_code != 200:
            self.logger.error(
                f"BGE Reranker API调用失败，状态码: {response.status_code}"
            )
            self.logger.error(f"响应内容: {response.text}")
            raise Exception(f"BGE Reranker API调用失败: {response.status_code}")

        result = response.json()

        if "error" in result:
            self.logger.error(f"BGE Reranker API返回错误: {result['error']}")
            raise Exception(f"BGE Reranker API错误: {result['error']}")

        # 解析BGE reranker结果
        return self._parse_bge_rerank_result(result, documents)

    def _rerank_with_chat_api(
        self, query: str, documents: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """使用Chat Completions API进行重排"""
        # 构建文档列表用于重排
        doc_list = []
        for i, doc in enumerate(documents):
            content = doc.get("content", doc.get("text", ""))
            if isinstance(content, str):
                # 截取前500字符避免token过多
                doc_list.append(f"[{i+1}] {content[:500]}")
            else:
                doc_list.append(f"[{i+1}] {str(content)[:500]}")

        # 构建重排提示
        prompt = self._build_rerank_prompt(query, doc_list)

        # 调用OpenAI API
        response = self._call_chat_api(prompt)

        # 解析重排结果
        return self._parse_chat_rerank_result(response, documents)

    def _parse_bge_rerank_result(
        self, api_result: Dict[str, Any], original_documents: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """解析BGE reranker API返回结果"""
        try:
            reranked_docs = []

            # BGE reranker通常返回results字段，包含index和score
            results = api_result.get("results", api_result.get("data", []))

            for item in results:
                if isinstance(item, dict):
                    index = item.get("index", item.get("document_index", -1))
                    score = item.get("relevance_score", item.get("score", 0.0))

                    if 0 <= index < len(original_documents):
                        doc = original_documents[index].copy()
                        doc["rerank_score"] = score
                        reranked_docs.append(doc)

            # 如果没有找到results字段，尝试其他可能的格式
            if not reranked_docs and "rankings" in api_result:
                rankings = api_result["rankings"]
                for i, ranking in enumerate(rankings[: self.top_k]):
                    if isinstance(ranking, dict):
                        index = ranking.get("index", i)
                        score = ranking.get("score", 1.0 - i * 0.1)
                    else:
                        index = ranking
                        score = 1.0 - i * 0.1

                    if 0 <= index < len(original_documents):
                        doc = original_documents[index].copy()
                        doc["rerank_score"] = score
                        reranked_docs.append(doc)

            return (
                reranked_docs[: self.top_k]
                if reranked_docs
                else original_documents[: self.top_k]
            )

        except Exception as e:
            self.logger.error(f"解析BGE重排结果失败: {e}")
            return original_documents[: self.top_k]

    def _build_rerank_prompt(self, query: str, doc_list: List[str]) -> str:
        """构建重排提示"""
        prompt = f"""请根据查询问题对以下文档按相关性从高到低重新排序。

查询问题：{query}

文档列表：
{chr(10).join(doc_list)}

请按相关性从高到低排序，只输出文档序号列表，格式如：[1,3,2,5,4]
不要输出其他内容，只要序号列表。"""
        return prompt

    def _call_chat_api(self, prompt: str) -> str:
        """调用Chat Completions API"""
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        response = requests.post(url, headers=headers, json=data, timeout=self.timeout)

        if response.status_code != 200:
            self.logger.error(f"OpenAI API调用失败，状态码: {response.status_code}")
            self.logger.error(f"响应内容: {response.text}")
            raise Exception(f"OpenAI API调用失败: {response.status_code}")

        result = response.json()

        if "error" in result:
            self.logger.error(f"OpenAI API返回错误: {result['error']}")
            raise Exception(f"OpenAI API错误: {result['error']}")

        return result["choices"][0]["message"]["content"]

    def _parse_chat_rerank_result(
        self, api_response: str, original_documents: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """解析Chat API返回结果"""
        try:
            import re

            # 提取序号列表
            match = re.search(r"\[(\d+(?:,\s*\d+)*)\]", api_response)
            if match:
                indices_str = match.group(1)
                indices = [int(x.strip()) - 1 for x in indices_str.split(",")]

                # 重新排序文档
                reranked_docs = []
                for idx in indices:
                    if 0 <= idx < len(original_documents):
                        doc = original_documents[idx].copy()
                        # 添加重排分数（基于排序位置）
                        doc["rerank_score"] = 1.0 - (len(reranked_docs) * 0.1)
                        reranked_docs.append(doc)

                # 添加未排序的文档
                used_indices = set(
                    idx for idx in indices if 0 <= idx < len(original_documents)
                )
                for i, doc in enumerate(original_documents):
                    if i not in used_indices and len(reranked_docs) < self.top_k:
                        doc_copy = doc.copy()
                        doc_copy["rerank_score"] = 0.1  # 给未排序文档较低分数
                        reranked_docs.append(doc_copy)

                return reranked_docs[: self.top_k]

            else:
                self.logger.warning(f"无法解析OpenAI重排结果: {api_response}")
                return original_documents[: self.top_k]

        except Exception as e:
            self.logger.error(f"解析OpenAI重排结果失败: {e}")
            return original_documents[: self.top_k]
