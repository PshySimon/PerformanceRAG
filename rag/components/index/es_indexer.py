from typing import Any, Dict, List, Optional

import urllib3
from elasticsearch.helpers import bulk

from .base_indexer import BaseIndexer

# 禁用 urllib3 的 SSL 警告
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

try:
    from elasticsearch import Elasticsearch
    from elasticsearch.exceptions import NotFoundError, RequestError
except ImportError:
    Elasticsearch = None
    NotFoundError = Exception
    RequestError = Exception


class ESIndexerComponent(BaseIndexer):
    """Elasticsearch索引器组件"""

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

        # 索引配置
        self.mapping = config.get("mapping", self._default_mapping())
        self.settings = config.get("settings", self._default_settings())

        self.client = None

    def _do_initialize(self):
        """初始化ES客户端"""
        try:
            # 构建连接配置
            if self.use_ssl:
                # 对于 ES 8.x，使用 https scheme
                hosts = [f"https://{self.host}:{self.port}"]
            else:
                hosts = [f"http://{self.host}:{self.port}"]

            es_config = {
                "hosts": hosts,
                "verify_certs": self.verify_certs,
            }

            if self.username and self.password:
                es_config["basic_auth"] = (self.username, self.password)

            # 移除 use_ssl 参数，因为 ES 8.x 不支持
            # if self.use_ssl:
            #     es_config["use_ssl"] = True

            self.client = Elasticsearch(**es_config)

            # 测试连接
            if not self.client.ping():
                raise ConnectionError("无法连接到Elasticsearch")

            if self.debug:
                self.logger.debug(f"成功连接到Elasticsearch: {self.host}:{self.port}")

        except Exception as e:
            self.logger.error(f"初始化Elasticsearch客户端失败: {e}")
            raise

    def _default_mapping(self) -> Dict[str, Any]:
        """默认映射配置"""
        return {
            "properties": {
                "content": {"type": "text", "analyzer": "standard"},
                "metadata": {"type": "object", "enabled": True},
                "timestamp": {"type": "date"},
            }
        }

    def _default_settings(self) -> Dict[str, Any]:
        """默认设置配置"""
        return {
            "number_of_shards": 1,
            "number_of_replicas": 0,
            "analysis": {"analyzer": {"default": {"type": "standard"}}},
        }

    def create_index(self, index_name: Optional[str] = None, **kwargs) -> bool:
        """创建索引"""
        target_index = index_name or self.index_name
        try:
            if self.client.indices.exists(index=target_index):
                if self.debug:
                    self.logger.debug(f"索引 {target_index} 已存在")
                return True

            body = {"mappings": self.mapping, "settings": self.settings}
            self.client.indices.create(index=target_index, body=body)

            if self.debug:
                self.logger.debug(f"成功创建索引: {target_index}")
            return True

        except Exception as e:
            self.logger.error(f"创建索引失败: {e}")
            return False

    # 移除 delete_index 方法 - 太危险了！
    # def delete_index(self) -> bool:
    #     """删除索引 - 已移除，太危险"""
    #     pass

    def index_documents(
        self, documents: List[Dict[str, Any]], index_name: Optional[str] = None
    ) -> bool:
        """批量索引文档"""
        target_index = index_name or self.index_name
        try:
            # 确保索引存在
            self.create_index(target_index)

            # 准备批量操作
            actions = []
            for i, doc in enumerate(documents):
                doc_id = doc.get("id", f"doc_{i}")
                action = {"_index": target_index, "_id": doc_id, "_source": doc}
                actions.append(action)

            success_count, failed_items = bulk(
                self.client, actions, chunk_size=self.batch_size, request_timeout=60
            )

            # 添加刷新操作，确保文档立即可搜索
            self.client.indices.refresh(index=target_index)

            if self.debug:
                self.logger.debug(f"成功索引 {success_count} 个文档到 {target_index}")
                if failed_items:
                    self.logger.warning(f"失败 {len(failed_items)} 个文档")

            return len(failed_items) == 0

        except Exception as e:
            self.logger.error(f"索引文档失败: {e}")
            return False

    def search(
        self, query: str, top_k: int = 10, index_name: Optional[str] = None, **kwargs
    ) -> List[Dict[str, Any]]:
        """搜索文档"""
        target_index = index_name or self.index_name
        try:
            # 构建查询 - 修复：明确指定搜索字段，避免搜索日期字段
            # 可以使用 bool 查询组合多个条件
            search_body = {
                "query": {
                    "bool": {
                        "should": [
                            {"match": {"content": {"query": query, "boost": 2}}},
                            {
                                "match": {
                                    "metadata.title": {"query": query, "boost": 1.5}
                                }
                            },
                            {"match": {"metadata.category": query}},
                            {"match": {"metadata.tags": query}},
                        ]
                    }
                },
                "size": top_k,
            }

            # 执行搜索
            response = self.client.search(index=target_index, body=search_body)

            # 处理结果
            results = []
            for hit in response["hits"]["hits"]:
                result = {
                    "id": hit["_id"],
                    "score": hit["_score"],
                    "content": hit["_source"].get("content", ""),
                    "metadata": hit["_source"].get("metadata", {}),
                }
                results.append(result)

            if self.debug:
                self.logger.debug(f"从 {target_index} 搜索返回 {len(results)} 个结果")

            return results

        except Exception as e:
            self.logger.error(f"搜索失败: {e}")
            return []

    def get_document(
        self, doc_id: str, index_name: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """根据ID获取文档"""
        target_index = index_name or self.index_name
        try:
            response = self.client.get(index=target_index, id=doc_id)
            return {
                "id": response["_id"],
                "content": response["_source"].get("content", ""),
                "metadata": response["_source"].get("metadata", {}),
            }
        except NotFoundError:
            return None
        except Exception as e:
            self.logger.error(f"获取文档失败: {e}")
            return None
