import uuid
from typing import Any, Dict, List

import requests

from .base_reranker import BaseRerankerComponent


class ZhipuRerankerComponent(BaseRerankerComponent):
    """智谱重排组件"""

    def __init__(self, name: str, config: Dict[str, Any]):
        super().__init__(name, config)
        self.api_key = config.get("api_key")
        self.api_url = config.get(
            "api_url", "https://open.bigmodel.cn/api/paas/v4/rerank"
        )
        self.return_documents = config.get("return_documents", True)
        self.return_raw_scores = config.get("return_raw_scores", True)
        self.timeout = config.get("timeout", 30)

        if not self.api_key:
            raise ValueError("智谱reranker需要配置api_key")

    def _do_initialize(self):
        """初始化智谱重排组件"""
        try:
            # 跳过连接测试，避免编码问题
            self.logger.info(f"智谱重排组件 {self.name} 初始化成功")
        except Exception as e:
            self.logger.error(f"智谱重排组件 {self.name} 初始化失败: {e}")
            raise

    def _rerank(
        self, query: str, documents: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """使用智谱API进行重排"""
        if not documents:
            return []

        try:
            # 准备文档内容，确保编码正确
            doc_texts = []
            for doc in documents:
                content = doc.get("content", doc.get("text", ""))
                if isinstance(content, str):
                    doc_texts.append(content)
                else:
                    doc_texts.append(str(content))

            # 构建API请求数据
            request_data = {
                "request_id": str(uuid.uuid4()),
                "query": query,
                "top_n": min(self.top_k, len(documents)),
                "documents": doc_texts,
                "return_documents": self.return_documents,
                "return_raw_scores": self.return_raw_scores,
            }

            headers = {
                "Authorization": self.api_key,  # 直接使用API密钥，不加Bearer前缀
                "User-Agent": "PerformanceRag/1.0.0",
                "Content-Type": "application/json; charset=utf-8",
            }

            # 调用智谱API
            response = requests.post(
                self.api_url, headers=headers, json=request_data, timeout=self.timeout
            )

            if response.status_code != 200:
                self.logger.error(f"智谱API调用失败，状态码: {response.status_code}")
                return documents[: self.top_k]

            result = response.json()

            # 解析API响应
            return self._parse_rerank_result(result, documents)

        except Exception as e:
            self.logger.error(f"智谱重排失败: {e}")
            return documents[: self.top_k]

    def _parse_rerank_result(
        self, api_result: Dict[str, Any], original_documents: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """解析智谱API返回结果"""
        try:
            # 检查API响应状态
            if "error" in api_result:
                self.logger.error(f"智谱API返回错误: {api_result['error']}")
                return original_documents[: self.top_k]

            # 获取重排结果
            results = api_result.get("results", [])
            if not results:
                self.logger.warning("智谱API返回空结果")
                return original_documents[: self.top_k]

            reranked_docs = []
            for item in results:
                # 获取文档索引
                index = item.get("index")
                if index is not None and 0 <= index < len(original_documents):
                    doc = original_documents[index].copy()

                    # 添加重排分数
                    if "relevance_score" in item:
                        doc["rerank_score"] = item["relevance_score"]

                    reranked_docs.append(doc)

            # 如果重排结果数量不足，补充原始文档
            if len(reranked_docs) < self.top_k:
                used_indices = {
                    item.get("index") for item in results if "index" in item
                }
                for i, doc in enumerate(original_documents):
                    if i not in used_indices and len(reranked_docs) < self.top_k:
                        reranked_docs.append(doc)

            return reranked_docs[: self.top_k]

        except Exception as e:
            self.logger.error(f"解析智谱API结果失败: {e}")
            return original_documents[: self.top_k]
