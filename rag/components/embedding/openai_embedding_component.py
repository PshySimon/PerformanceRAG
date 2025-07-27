from typing import Any, Dict

from ..base import Component
from .embedding_factory import EmbeddingFactory


class OpenAIEmbeddingComponent(Component):
    """OpenAI Embedding组件"""

    def __init__(self, name: str, config: Dict[str, Any]):
        super().__init__(name, config)
        self.embedding_client = None

    def _do_initialize(self):
        """初始化embedding客户端"""
        try:
            # 从配置创建embedding客户端
            embedding_config = self.config.copy()
            embedding_config["type"] = "openai"
            self.embedding_client = EmbeddingFactory.from_config(embedding_config)

            if self.debug:
                self.logger.debug(
                    f"OpenAI Embedding组件初始化完成，维度: {self.embedding_client.get_dimensions()}"
                )

        except Exception as e:
            self.logger.error(f"OpenAI Embedding组件初始化失败: {e}")
            raise

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """处理文档，添加向量化结果"""
        try:
            documents = data.get("documents", [])

            if not documents:
                return data

            # 提取文档内容
            texts = []
            for doc in documents:
                content = doc.get("content", doc.get("text", ""))
                texts.append(content)

            # 批量向量化
            embeddings = self.embedding_client.embed_texts(texts)

            # 将向量添加到文档中
            for doc, embedding in zip(documents, embeddings):
                doc["content_vector"] = embedding

            if self.debug:
                self.logger.debug(f"成功向量化 {len(documents)} 个文档")

            return data

        except Exception as e:
            self.logger.error(f"文档向量化失败: {e}")
            raise
