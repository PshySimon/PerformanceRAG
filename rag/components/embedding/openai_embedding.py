import os
from typing import List

import openai

from .base_embedding import BaseEmbedding


class OpenAIEmbedding(BaseEmbedding):
    """OpenAI embedding客户端，支持单条和批量文本embedding"""

    def __init__(
        self,
        model: str,
        api_key: str,
        api_base: str,
        batch_size: int,
        timeout: int = 60,
        max_retries: int = 3,
    ):
        """初始化OpenAI embedding客户端

        Args:
            model: 使用的嵌入模型名称
            api_key: OpenAI API密钥，如果为None则从环境变量获取
            api_base: OpenAI API基础URL，如果为None则使用默认值
            batch_size: 批处理大小
            timeout: 请求超时时间（秒）
            max_retries: 最大重试次数
        """
        # 设置API密钥
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API密钥未设置，请通过参数传入或设置OPENAI_API_KEY环境变量"
            )

        # 设置API基础URL
        self.api_base = api_base

        # 设置模型和批处理参数
        self.model = model
        self.batch_size = batch_size
        self.timeout = timeout
        self.max_retries = max_retries

        # 初始化OpenAI客户端
        self._init_client()

    def _init_client(self):
        """初始化OpenAI客户端"""
        client_kwargs = {
            "api_key": self.api_key,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
        }

        if self.api_base:
            client_kwargs["base_url"] = self.api_base

        self.client = openai.OpenAI(**client_kwargs)

    def embed_text(self, text: str) -> List[float]:
        """对单条文本进行embedding

        Args:
            text: 需要嵌入的文本

        Returns:
            List[float]: 嵌入向量
        """
        if not text.strip():
            return [0.0]  # 返回空向量

        response = self.client.embeddings.create(model=self.model, input=text)

        return response.data[0].embedding

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """对多条文本进行批量embedding

        Args:
            texts: 需要嵌入的文本列表

        Returns:
            List[List[float]]: 嵌入向量列表
        """
        if not texts:
            return []  # 返回空列表

        # 过滤空文本
        filtered_texts = [text for text in texts if text.strip()]
        if not filtered_texts:
            return [[0.0]] * len(texts)  # 全部是空文本，返回空向量列表

        results = []
        # 批量处理
        for i in range(0, len(filtered_texts), self.batch_size):
            batch = filtered_texts[i : i + self.batch_size]
            response = self.client.embeddings.create(model=self.model, input=batch)
            batch_embeddings = [data.embedding for data in response.data]
            results.extend(batch_embeddings)

        # 处理原始文本中的空文本
        final_results = []
        result_idx = 0
        for text in texts:
            if text.strip():
                final_results.append(results[result_idx])
                result_idx += 1
            else:
                final_results.append([0.0])  # 空文本返回空向量

        return final_results
