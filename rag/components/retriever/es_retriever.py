from typing import Any, Dict, List, Optional

from elasticsearch import Elasticsearch

from utils.config import config
from ..embedding.embedding_factory import EmbeddingFactory

from .base_retriever import BaseRetrieverComponent


class ESRetrieverComponent(BaseRetrieverComponent):
    """Elasticsearch检索器组件"""

    def __init__(self, name: str, config: Dict[str, Any]):
        super().__init__(name, config)

        if Elasticsearch is None:
            raise ImportError("请安装elasticsearch包: pip install elasticsearch")

        # ES配置
        self.host = config.get("host", "localhost")
        self.port = config.get("port", 9200)
        self.username = config.get("username")
        self.password = config.get("password")
        self.use_ssl = config.get("use_ssl", False)
        self.verify_certs = config.get("verify_certs", False)

        # 检索配置
        self.search_type = config.get("search_type", "text")  # text, vector, hybrid
        self.embedding_config = config.get("embedding", {})

        # 内部状态
        self.client = None
        self.embedding_client = None

    def _do_initialize(self):
        """初始化ES客户端和embedding客户端"""
        try:
            self._connect_to_index()

            # 如果需要向量检索，初始化embedding客户端
            if self.search_type in ["vector", "hybrid"]:
                self._init_embedding_client()

            if self.debug:
                self.logger.debug(f"ES检索器初始化完成，检索类型: {self.search_type}")

        except Exception as e:
            self.logger.error(f"初始化ES检索器失败: {e}")
            raise

    def _connect_to_index(self):
        """连接到ES索引"""
        try:
            # 构建连接配置
            if self.use_ssl:
                hosts = [f"https://{self.host}:{self.port}"]
            else:
                hosts = [f"http://{self.host}:{self.port}"]

            es_config = {
                "hosts": hosts,
                "verify_certs": self.verify_certs,
            }

            if self.username and self.password:
                es_config["basic_auth"] = (self.username, self.password)

            self.client = Elasticsearch(**es_config)

            # 测试连接
            if not self.client.ping():
                raise ConnectionError("无法连接到Elasticsearch")

            if self.debug:
                self.logger.debug(f"成功连接到Elasticsearch: {self.host}:{self.port}")

        except Exception as e:
            self.logger.error(f"连接Elasticsearch失败: {e}")
            raise

    def _init_embedding_client(self):
        """初始化embedding客户端"""
        try:
            embedding_type = self.embedding_config.get("type", "hf")

            if embedding_type == "hf":
                emb_cfg = config.embeddings.clients.hf
                self.embedding_client = EmbeddingFactory.create(
                    "hf", model_name=emb_cfg.model_name
                )
            elif embedding_type == "openai":
                emb_cfg = config.embeddings.clients.openai
                self.embedding_client = EmbeddingFactory.create(
                    "openai", api_key=emb_cfg.api_key
                )
            elif embedding_type == "bge_embedding":
                emb_cfg = config.embeddings.clients.bge_embedding
                self.embedding_client = EmbeddingFactory.create(
                    "bge_embedding", api_url=emb_cfg.api_url
                )
            else:
                raise ValueError(f"不支持的embedding类型: {embedding_type}")

            if self.debug:
                self.logger.debug(f"Embedding客户端初始化完成: {embedding_type}")

        except Exception as e:
            self.logger.error(f"初始化embedding客户端失败: {e}")
            raise

    def retrieve(
        self, query: str, top_k: Optional[int] = None, **kwargs
    ) -> List[Dict[str, Any]]:
        """检索文档"""
        use_top_k = top_k or self.top_k

        try:
            if self.search_type == "text":
                return self._text_search(query, use_top_k)
            elif self.search_type == "vector":
                return self._vector_search(query, use_top_k)
            elif self.search_type == "hybrid":
                return self._hybrid_search(query, use_top_k)
            else:
                raise ValueError(f"不支持的检索类型: {self.search_type}")

        except Exception as e:
            self.logger.error(f"检索失败: {e}")
            return []

    def _text_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """文本检索"""
        search_body = {
            "query": {
                "bool": {
                    "should": [
                        {"match": {"content": {"query": query, "boost": 2}}},
                        {"match": {"metadata.title": {"query": query, "boost": 1.5}}},
                        {"match": {"metadata.category": query}},
                        {"match": {"metadata.tags": query}},
                    ]
                }
            },
            "size": top_k,
        }

        response = self.client.search(index=self.index_name, body=search_body)
        return self._format_results(response)

    def _vector_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """向量检索"""
        if not self.embedding_client:
            raise RuntimeError("向量检索需要embedding客户端")

        # 生成查询向量
        query_embedding = self.embedding_client.embed_text(query)

        # KNN检索
        search_body = {
            "knn": {
                "field": "embedding",
                "query_vector": query_embedding,
                "k": top_k,
                "num_candidates": top_k * 2,
            },
            "size": top_k,
        }

        response = self.client.search(index=self.index_name, body=search_body)
        return self._format_results(response)

    def _hybrid_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """混合检索（文本+向量）"""
        if not self.embedding_client:
            raise RuntimeError("混合检索需要embedding客户端")

        # 生成查询向量
        query_embedding = self.embedding_client.embed_text(query)

        # 混合查询
        search_body = {
            "query": {
                "bool": {
                    "should": [
                        {"match": {"content": {"query": query, "boost": 1}}},
                        {"match": {"metadata.title": {"query": query, "boost": 0.8}}},
                    ]
                }
            },
            "knn": {
                "field": "embedding",
                "query_vector": query_embedding,
                "k": top_k,
                "num_candidates": top_k * 2,
                "boost": 1.2,
            },
            "size": top_k,
        }

        response = self.client.search(index=self.index_name, body=search_body)
        return self._format_results(response)

    def _format_results(self, response) -> List[Dict[str, Any]]:
        """格式化检索结果"""
        results = []
        for hit in response["hits"]["hits"]:
            result = {
                "id": hit["_id"],
                "score": hit["_score"],
                "content": hit["_source"].get("content", ""),
                "metadata": hit["_source"].get("metadata", {}),
            }

            # 过滤低分结果
            if result["score"] >= self.similarity_threshold:
                results.append(result)

        if self.debug:
            self.logger.debug(f"ES检索返回 {len(results)} 个结果")

        return results
