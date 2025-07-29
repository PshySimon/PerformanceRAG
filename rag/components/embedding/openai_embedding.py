import os
from typing import Any, Dict, List

import openai

from ..base import Component
from .base_embedding import BaseEmbedding
from utils.logger import get_logger  # 👈 添加logger导入


class OpenAIEmbedding(BaseEmbedding):
    """OpenAI embedding客户端，支持单条和批量文本embedding"""

    def __init__(
        self,
        model: str,
        api_key: str,
        api_base: str,
        batch_size: int,
        dimensions: int = None,
        timeout: int = 60,
        max_retries: int = 3,
    ):
        """初始化OpenAI embedding客户端

        Args:
            model: 使用的嵌入模型名称
            api_key: OpenAI API密钥，如果为None则从环境变量获取
            api_base: OpenAI API基础URL，如果为None则使用默认值
            batch_size: 批处理大小
            dimensions: 嵌入向量维度，可选参数
            timeout: 请求超时时间（秒）
            max_retries: 最大重试次数
        """
        # 移除强制要求dimensions参数的检查
        super().__init__(dimensions=dimensions)
        
        # 👈 添加logger初始化
        self.logger = get_logger(f"{__name__}.OpenAIEmbedding")
        
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
        if not text or not text.strip():
            raise ValueError(f"不能对空文本进行向量化，输入文本: '{text}'")
    
        # 记录调试信息
        if self.logger:
            self.logger.debug(f"正在向量化文本: '{text[:100]}...' (长度: {len(text)})")
            self.logger.debug(f"使用模型: {self.model}, API基础URL: {self.api_base}")
    
        # 简单的请求参数，不管dimensions
        request_params = {
            "model": self.model,
            "input": text
        }
    
        try:
            response = self.client.embeddings.create(**request_params)
            
            # 检查响应数据
            if not response.data:
                raise ValueError(f"OpenAI API 返回空数据，模型: {self.model}, 文本长度: {len(text)}")
            
            if not response.data[0].embedding:
                raise ValueError(f"OpenAI API 返回空向量，模型: {self.model}, 响应: {response}")
            
            embedding = response.data[0].embedding
            if self.logger:
                self.logger.debug(f"成功获取向量，维度: {len(embedding)}")
            
            return embedding
            
        except openai.APIStatusError as e:
            # 捕获HTTP状态码错误
            error_msg = f"OpenAI API 返回非200状态码: {e.status_code}\n" \
                       f"错误类型: {e.type}\n" \
                       f"错误消息: {e.message}\n" \
                       f"模型: {self.model}\n" \
                       f"API基础URL: {self.api_base}\n" \
                       f"文本长度: {len(text)}\n" \
                       f"文本内容: '{text[:200]}...'"
            if self.logger:
                self.logger.error(error_msg)
            raise ValueError(error_msg) from e
        except openai.APIConnectionError as e:
            # 捕获连接错误
            error_msg = f"OpenAI API 连接失败: {e}\n" \
                       f"模型: {self.model}\n" \
                       f"API基础URL: {self.api_base}\n" \
                       f"文本长度: {len(text)}\n" \
                       f"文本内容: '{text[:200]}...'"
            if self.logger:
                self.logger.error(error_msg)
            raise ValueError(error_msg) from e
        except Exception as e:
            error_msg = f"OpenAI embedding 调用失败: {e}\n" \
                       f"模型: {self.model}\n" \
                       f"API基础URL: {self.api_base}\n" \
                       f"文本长度: {len(text)}\n" \
                       f"文本内容: '{text[:200]}...'"
            if self.logger:
                self.logger.error(error_msg)
            raise ValueError(error_msg) from e

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """对多条文本进行批量embedding
    
        Args:
            texts: 需要嵌入的文本列表
    
        Returns:
            List[List[float]]: 嵌入向量列表
        """
        if not texts:
            return []
    
        # 过滤掉空文本，直接忽略
        filtered_texts = [text for text in texts if text.strip()]
        
        if not filtered_texts:
            return []
        
        results = []
        # 批量处理
        for i in range(0, len(filtered_texts), self.batch_size):
            batch = filtered_texts[i:i + self.batch_size]
            
            # 简单的请求参数，不管dimensions
            request_params = {
                "model": self.model,
                "input": batch
            }
    
            try:
                response = self.client.embeddings.create(**request_params)
                batch_embeddings = [data.embedding for data in response.data]
                results.extend(batch_embeddings)
            except openai.APIStatusError as e:
                # 捕获HTTP状态码错误
                error_msg = f"批量embedding请求返回非200状态码: {e.status_code}\n" \
                           f"错误类型: {e.type}\n" \
                           f"错误消息: {e.message}\n" \
                           f"模型: {self.model}\n" \
                           f"API基础URL: {self.api_base}\n" \
                           f"批次大小: {len(batch)}"
                if self.logger:
                    self.logger.error(error_msg)
                raise ValueError(error_msg) from e
            except openai.APIConnectionError as e:
                # 捕获连接错误
                error_msg = f"批量embedding请求连接失败: {e}\n" \
                           f"模型: {self.model}\n" \
                           f"API基础URL: {self.api_base}\n" \
                           f"批次大小: {len(batch)}"
                if self.logger:
                    self.logger.error(error_msg)
                raise ValueError(error_msg) from e
            except Exception as e:
                error_msg = f"批量embedding请求失败: {e}\n" \
                           f"模型: {self.model}\n" \
                           f"API基础URL: {self.api_base}\n" \
                           f"批次大小: {len(batch)}"
                if self.logger:
                    self.logger.error(error_msg)
                raise ValueError(error_msg) from e
    
        return results


class OpenAIEmbeddingComponent(Component):
    """OpenAI Embedding组件"""

    def __init__(self, name: str, config: Dict[str, Any]):
        super().__init__(name, config)
        self.embedding_client = None
        # 👈 添加缓存机制
        self.text_cache = []  # 缓存待处理的文本
        self.doc_cache = []   # 缓存对应的文档对象
        self.cache_size = config.get("batch_size", 100)  # 缓存大小
        self.force_flush = False  # 强制刷新标志

    def _do_initialize(self):
        """初始化embedding客户端"""
        try:
            # 从配置创建embedding客户端
            self.embedding_client = OpenAIEmbedding(
                model=self.config.get("model"),
                api_key=self.config.get("api_key"),
                api_base=self.config.get("api_base"),
                batch_size=self.config.get("batch_size", 10),
                dimensions=self.config.get("dimensions"),  # 允许为None
                timeout=self.config.get("timeout", 60),
                max_retries=self.config.get("max_retries", 3),
            )

            if self.debug:
                dims = self.embedding_client.get_dimensions()
                if dims is not None:
                    self.logger.debug(
                        f"OpenAI Embedding组件初始化完成，维度: {dims}"
                    )
                else:
                    self.logger.debug(
                        "OpenAI Embedding组件初始化完成，维度将在首次调用时确定"
                    )

        except Exception as e:
            self.logger.error(f"OpenAI Embedding组件初始化失败: {e}")
            raise

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """处理文档，添加向量化结果（带缓存机制）"""
        try:
            documents = data.get("documents", [])
    
            if not documents:
                return data
    
            # 将新文档添加到缓存
            for doc in documents:
                content = doc.get("content", doc.get("text", ""))
                self.text_cache.append(content)
                self.doc_cache.append(doc)
    
            # 检查是否需要处理缓存
            if len(self.text_cache) >= self.cache_size or self.force_flush:
                return self._process_cached_texts()
            else:
                return {"documents": []}
    
        except Exception as e:
            self.logger.error(f"❌ 文档向量化失败: {e}")
            raise

    def _process_cached_texts(self) -> Dict[str, Any]:
        """处理缓存中的文本"""
        if not self.text_cache:
            return {"documents": []}

        # 批量向量化
        embeddings = self.embedding_client.embed_texts(self.text_cache)

        # 将向量添加到对应文档中
        for doc, embedding in zip(self.doc_cache, embeddings):
            doc["content_vector"] = embedding

        # 准备返回数据
        result_docs = self.doc_cache.copy()
        
        # 清空缓存
        self.text_cache.clear()
        self.doc_cache.clear()
        self.force_flush = False
        return {"documents": result_docs}

    def flush_cache(self):
        """强制刷新缓存（在pipeline结束时调用）"""
        self.force_flush = True
        if self.text_cache:
            return self._process_cached_texts()
        return {"documents": []}

