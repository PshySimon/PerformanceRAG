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
            
        # 优先检索的chunk_level
        self.preferred_chunk_level = config.get("preferred_chunk_level")

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
        
        # Small2Big检索配置
        self.enable_small2big = config.get("enable_small2big", False)
        small2big_config = config.get("small2big_config", {})
        self.small_chunk_top_k = small2big_config.get("small_chunk_top_k", 20)
        self.final_top_k = small2big_config.get("final_top_k", 10)
        self.expansion_strategy = small2big_config.get("expansion_strategy", "parent_expansion")
        self.similarity_threshold_small = small2big_config.get("similarity_threshold_small", 0.6)
        self.score_fusion_method = small2big_config.get("score_fusion_method", "max")
        self.enable_score_normalization = small2big_config.get("enable_score_normalization", True)
        self.diversity_threshold = small2big_config.get("diversity_threshold", 0.8)
        self.max_small_chunks_per_big = small2big_config.get("max_small_chunks_per_big", 5)
        
        # RRF配置
        self.fusion_method = config.get("fusion_method", "weighted")  # weighted, rrf
        self.rrf_k = config.get("rrf_k", 60)  # RRF参数
        self.text_weight = config.get("text_weight", 1.0)  # 文本检索权重
        self.vector_weight = config.get("vector_weight", 1.2)  # 向量检索权重

        # 添加自定义查询字段支持
        self.search_fields = config.get(
            "search_fields",
            {
                "content": 2.0,
                "metadata.title": 1.5,
                "metadata.category": 1.0,
                "metadata.tags": 1.0,
            },
        )
        
        # 添加分析器配置读取
        self.analyzer_config = config.get("analyzer_config", {})
        self.search_analyzer = self.analyzer_config.get("search_analyzer", "ik_search_analyzer")
        self.index_analyzer = self.analyzer_config.get("index_analyzer", "ik_analyzer")
        
        # 添加高亮配置读取
        self.highlight_config = config.get("highlight_config", {})
        self.default_highlight_settings = {
            "require_field_match": self.highlight_config.get("require_field_match", False),
            "fragment_size": self.highlight_config.get("fragment_size", 150),
            "number_of_fragments": self.highlight_config.get("number_of_fragments", 3),
            "pre_tags": self.highlight_config.get("pre_tags", ["<mark>"]),
            "post_tags": self.highlight_config.get("post_tags", ["</mark>"]),
        }

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
                self.logger.debug(f"ES检索器初始化完成，检索类型: {self.search_type}, 融合方法: {self.fusion_method}")

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
                # 直接使用配置文件中的参数
                self.embedding_client = EmbeddingFactory.create(
                    "openai",
                    model=self.embedding_config.get("model"),
                    api_key=self.embedding_config.get("api_key"),
                    api_base=self.embedding_config.get("api_base"),
                    batch_size=self.embedding_config.get("batch_size", 10),
                    dimensions=self.embedding_config.get("dimensions"),
                    timeout=self.embedding_config.get("timeout", 60),
                    max_retries=self.embedding_config.get("max_retries", 3),
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
            if self.debug:
                self.logger.debug(
                    f"🔍 开始ES检索 - 查询: '{query}', 检索类型: {self.search_type}, Top-K: {use_top_k}"
                )

            # 如果启用Small2Big检索
            if self.enable_small2big:
                return self._small2big_search(query, use_top_k)
            
            # 原有检索逻辑
            if self.search_type == "text":
                return self._text_search(query, use_top_k)
            elif self.search_type == "vector":
                return self._vector_search(query, use_top_k)
            elif self.search_type == "hybrid":
                return self._hybrid_search(query, use_top_k)
            else:
                raise ValueError(f"不支持的检索类型: {self.search_type}")

        except Exception as e:
            import traceback

            self.logger.error(f"检索失败: {e}\n{traceback.format_exc()}")
            return []

    def _small2big_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Small2Big检索策略实现"""
        if self.debug:
            self.logger.info(f"🔍 开始Small2Big检索 - 查询: '{query}'")
            self.logger.info(f"   第一阶段: 检索Small Chunk (目标数量: {self.small_chunk_top_k})")
        
        # 第一阶段：检索Small Chunk
        small_chunks = self._search_small_chunks(query, self.small_chunk_top_k)
        
        if not small_chunks:
            self.logger.warning("⚠️ Small Chunk检索无结果")
            return []
        
        if self.debug:
            self.logger.info(f"   ✅ Small Chunk检索完成，获得 {len(small_chunks)} 个结果")
            self.logger.info(f"   第二阶段: 扩展到Big Chunk并融合分数")
        
        # 第二阶段：根据策略扩展到Big Chunk
        big_chunks = self._expand_to_big_chunks(small_chunks)
        
        if self.debug:
            self.logger.info(f"   ✅ 扩展完成，获得 {len(big_chunks)} 个Big Chunk")
            self.logger.info(f"   第三阶段: 分数融合和重排序")
        
        # 第三阶段：分数融合和重排序
        final_results = self._fuse_and_rerank(big_chunks, top_k)
        
        if self.debug:
            self.logger.info(f"🎯 Small2Big检索完成，最终返回 {len(final_results)} 个结果")
            for i, result in enumerate(final_results[:5], 1):  # 显示前5个结果
                self.logger.debug(f"   #{i} ID: {result['id']}, 分数: {result['score']:.4f}, 来源Small Chunk数: {result.get('source_small_chunks_count', 1)}")
        
        return final_results
    
    def _build_highlight_fields(self) -> Dict[str, Any]:
        """动态构建高亮字段配置"""
        highlight_fields = {}
        
        # 从配置中获取字段特定的高亮设置
        field_configs = self.highlight_config.get("fields", {})
        
        # 为所有搜索字段添加高亮配置
        for field in self.search_fields.keys():
            if field in field_configs:
                # 使用字段特定配置
                field_config = field_configs[field].copy()
                highlight_fields[field] = field_config
            else:
                # 使用默认配置
                highlight_fields[field] = {
                    "fragment_size": self.default_highlight_settings["fragment_size"],
                    "number_of_fragments": self.default_highlight_settings["number_of_fragments"],
                    "pre_tags": self.default_highlight_settings["pre_tags"],
                    "post_tags": self.default_highlight_settings["post_tags"]
                }
        
        # 特别处理 content_jieba 字段，确保使用正确的分析器
        if "content_jieba" in highlight_fields:
            highlight_fields["content_jieba"]["analyzer"] = self.search_analyzer
            
        if self.debug:
            self.logger.debug(f"🎨 构建的高亮字段配置: {highlight_fields}")
            
        return highlight_fields

    def _search_small_chunks(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """检索Small Chunk - 动态检测版本"""
        # 动态检测chunk_level字段路径和值
        chunk_level_info = self._detect_chunk_level_config()
        
        if not chunk_level_info:
            self.logger.warning("无法检测chunk_level配置，使用默认值")
            chunk_level_filter = {
                "term": {
                    "metadata.chunk_level": 2  # 回退到默认值
                }
            }
        else:
            chunk_level_filter = {
                "term": {
                    chunk_level_info["field_path"]: chunk_level_info["small_chunk_level"]
                }
            }
        
        if self.debug:
            self.logger.debug(f"🔍 Small Chunk过滤条件: {chunk_level_filter}")

        
        if self.search_type == "hybrid":
            # 混合检索Small Chunk
            return self._hybrid_search_with_filter(query, top_k, chunk_level_filter)
        elif self.search_type == "vector":
            return self._vector_search_with_filter(query, top_k, chunk_level_filter)
        else:
            return self._text_search_with_filter(query, top_k, chunk_level_filter)

    def _expand_to_big_chunks(self, small_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """将Small Chunk扩展到Big Chunk"""
        big_chunk_groups = {}
        
        for small_chunk in small_chunks:
            metadata = small_chunk.get("metadata", {})
            
            if self.expansion_strategy == "parent_expansion":
                # 通过parent_id扩展
                parent_id = metadata.get("parent_id")
                if parent_id:
                    if parent_id not in big_chunk_groups:
                        big_chunk_groups[parent_id] = {
                            "small_chunks": [],
                            "max_score": 0,
                            "avg_score": 0,
                            "big_chunk_data": None
                        }
                    
                    big_chunk_groups[parent_id]["small_chunks"].append(small_chunk)
                    big_chunk_groups[parent_id]["max_score"] = max(
                        big_chunk_groups[parent_id]["max_score"], 
                        small_chunk["score"]
                    )
            
            elif self.expansion_strategy == "root_expansion":
                # 通过root_id扩展到最大粒度
                root_id = metadata.get("root_id")
                if root_id:
                    if root_id not in big_chunk_groups:
                        big_chunk_groups[root_id] = {
                            "small_chunks": [],
                            "max_score": 0,
                            "avg_score": 0,
                            "big_chunk_data": None
                        }
                    
                    big_chunk_groups[root_id]["small_chunks"].append(small_chunk)
                    big_chunk_groups[root_id]["max_score"] = max(
                        big_chunk_groups[root_id]["max_score"], 
                        small_chunk["score"]
                    )
        
        # 获取Big Chunk的完整内容
        for big_chunk_id, group_data in big_chunk_groups.items():
            big_chunk_content = self._get_big_chunk_content(big_chunk_id)
            if big_chunk_content:
                group_data["big_chunk_data"] = big_chunk_content
                
                # 计算平均分数
                scores = [chunk["score"] for chunk in group_data["small_chunks"]]
                group_data["avg_score"] = sum(scores) / len(scores)
        
        return big_chunk_groups

    def _get_big_chunk_content(self, big_chunk_id: str) -> Optional[Dict[str, Any]]:
        """获取Big Chunk的完整内容"""
        try:
            # 使用 .keyword 字段进行精确匹配
            search_body = {
                "query": {
                    "term": {
                        "metadata.chunk_id.keyword": big_chunk_id
                    }
                },
                "size": 1
            }
            
            response = self.client.search(index=self.index_name, body=search_body)
            
            if response["hits"]["total"]["value"] > 0:
                hit = response["hits"]["hits"][0]
                return {
                    "id": hit["_id"],
                    "content": hit["_source"].get("content", ""),
                    "metadata": hit["_source"].get("metadata", {})
                }
        except Exception as e:
            if self.debug:
                self.logger.warning(f"获取Big Chunk {big_chunk_id} 失败: {e}")
        return None

    def _fuse_and_rerank(self, big_chunk_groups: Dict, top_k: int) -> List[Dict[str, Any]]:
        """融合分数并重新排序"""
        results = []
        
        for big_chunk_id, group_data in big_chunk_groups.items():
            if not group_data["big_chunk_data"]:
                continue
            
            # 根据融合方法计算最终分数
            if self.score_fusion_method == "max":
                final_score = group_data["max_score"]
            elif self.score_fusion_method == "avg":
                final_score = group_data["avg_score"]
            elif self.score_fusion_method == "weighted_avg":
                # 加权平均，分数越高权重越大
                scores = [chunk["score"] for chunk in group_data["small_chunks"]]
                weights = [score / sum(scores) for score in scores]
                final_score = sum(score * weight for score, weight in zip(scores, weights))
            else:
                final_score = group_data["max_score"]
            
            result = {
                "id": big_chunk_id,
                "score": final_score,
                "content": group_data["big_chunk_data"]["content"],
                "metadata": group_data["big_chunk_data"]["metadata"],
                "recall_source": "small2big",
                "source_small_chunks": group_data["small_chunks"],
                "source_small_chunks_count": len(group_data["small_chunks"]),
                "max_small_score": group_data["max_score"],
                "avg_small_score": group_data["avg_score"],
                "fusion_method": self.score_fusion_method
            }
            
            results.append(result)
        
        # 按分数排序并返回top_k
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def _hybrid_search_with_filter(self, query: str, top_k: int, filter_clause: Dict) -> List[Dict[str, Any]]:
        """带过滤条件的混合检索"""
        if not self.embedding_client:
            raise RuntimeError("混合检索需要embedding客户端")
        
        vector_field = self.embedding_config.get("field_name", "embedding")
        query_embedding = self.embedding_client.embed_text(query)
        
        # 文本检索部分
        should_queries = []
        for field, boost in self.search_fields.items():
            should_queries.append({"match": {field: {"query": query, "boost": boost, "analyzer": self.search_analyzer}}})
        
        filter_clauses = [filter_clause]
        if preferred_chunk_level := self.config.get("preferred_chunk_level"):
            filter_clauses.append({
                "term": {
                    "metadata.chunk_level": preferred_chunk_level
                }
            })
        
        text_search_body = {
            "query": {
                "bool": {
                    "should": should_queries,
                    "filter": filter_clauses
                }
            },
            "size": top_k * 2,
        }
        
        # 向量检索部分
        vector_search_body = {
            "knn": {
                "field": vector_field,
                "query_vector": query_embedding,
                "k": top_k * 2,
                "num_candidates": top_k * 4,
                "filter": {
                    "bool": {
                        "filter": filter_clauses
                    }
                }
            },
            "size": top_k * 2,
        }
        
        text_response = self.client.search(index=self.index_name, body=text_search_body)
        vector_response = self.client.search(index=self.index_name, body=vector_search_body)
        
        # 使用现有的融合方法
        if self.fusion_method == "rrf":
            return self._merge_hybrid_results_with_rrf(text_response, vector_response, query, top_k)
        else:
            return self._merge_hybrid_results_with_highlights(text_response, vector_response, query, top_k)

    def _text_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """文本检索"""
        # 动态构建查询字段
        should_queries = []
        
        for field, boost in self.search_fields.items():
            should_queries.append({"match": {field: {"query": query, "boost": boost, "analyzer": self.search_analyzer}}})
        
        # 🔧 使用新的高亮字段构建方法
        highlight_fields = self._build_highlight_fields()
        
        search_body = {
            "query": {"bool": {"should": should_queries}},
            "highlight": {
                "fields": highlight_fields,
                "require_field_match": self.default_highlight_settings["require_field_match"]
            },
            "size": top_k,
        }
        
        if self.debug:
            self.logger.debug(f"📝 文本检索查询体: {search_body}")
        
        response = self.client.search(index=self.index_name, body=search_body)
        return self._format_results_with_highlights(response, query, "文本检索")

    def _vector_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """向量检索"""
        if not self.embedding_client:
            raise RuntimeError("向量检索需要embedding客户端")

        # 获取向量字段名
        vector_field = self.embedding_config.get("field_name", "embedding")
        
        # 生成查询向量
        if self.debug:
            self.logger.debug(f"🔢 正在生成向量检索查询向量: '{query}'")

        query_embedding = self.embedding_client.embed_text(query)

        if self.debug:
            self.logger.debug(
                f"✅ 向量检索查询向量生成完成，维度: {len(query_embedding)}"
            )

        # 向量检索
        vector_search_body = {
            "knn": {
                "field": vector_field,
                "query_vector": query_embedding,
                "k": top_k,
                "num_candidates": top_k * 4,
            },
            "size": top_k,
        }

        if self.debug:
            self.logger.debug(f"🎯 向量检索查询体: {vector_search_body}")

        response = self.client.search(index=self.index_name, body=vector_search_body)
        return self._format_results_with_highlights(response, query, "向量检索")

    def _text_search_with_filter(self, query: str, top_k: int, filter_clause: Dict) -> List[Dict[str, Any]]:
        """带过滤器的文本检索"""
        # 动态构建查询字段
        should_queries = []
        
        for field, boost in self.search_fields.items():
            should_queries.append({"match": {field: {"query": query, "boost": boost, "analyzer": self.search_analyzer}}})

        # 使用动态高亮字段构建方法
        highlight_fields = self._build_highlight_fields()

        search_body = {
            "query": {
                "bool": {
                    "should": should_queries,
                    "filter": filter_clause
                }
            },
            "highlight": {
                "fields": highlight_fields,
                "require_field_match": self.highlight_config.get("require_field_match", False)
            },
            "size": top_k,
        }

        if self.debug:
            self.logger.debug(f"📝 带过滤器的文本检索查询体: {search_body}")

        response = self.client.search(index=self.index_name, body=search_body)
        return self._format_results_with_highlights(response, query, "文本检索")

    def _detect_chunk_level_config(self) -> Optional[Dict[str, Any]]:
        """动态检测chunk_level配置"""
        try:
            # 检测字段路径
            possible_paths = ['chunk_level', 'metadata.chunk_level']
            
            for path in possible_paths:
                test_query = {
                    "query": {"exists": {"field": path}},
                    "size": 1
                }
                
                result = self.client.search(index=self.index_name, body=test_query)
                if result['hits']['total']['value'] > 0:
                    # 获取该字段的所有值
                    agg_query = {
                        "size": 0,
                        "aggs": {
                            "levels": {
                                "terms": {"field": path, "size": 10}
                            }
                        }
                    }
                    
                    agg_result = self.client.search(index=self.index_name, body=agg_query)
                    levels = [bucket['key'] for bucket in agg_result['aggregations']['levels']['buckets']]
                    
                    if levels:
                        # 选择最大的level作为small chunk level
                        small_chunk_level = max(levels)
                        
                        return {
                            "field_path": path,
                            "small_chunk_level": small_chunk_level,
                            "available_levels": levels
                        }
            
            return None
            
        except Exception as e:
            if self.debug:
                self.logger.error(f"检测chunk_level配置失败: {e}")
            return None

    def _vector_search_with_filter(self, query: str, top_k: int, filter_clause: Dict) -> List[Dict[str, Any]]:
        """带过滤器的向量检索"""
        if not self.embedding_client:
            raise RuntimeError("向量检索需要embedding客户端")

        # 获取向量字段名
        vector_field = self.embedding_config.get("field_name", "embedding")
        
        # 生成查询向量
        if self.debug:
            self.logger.debug(f"🔢 正在生成向量检索查询向量: '{query}'")

        query_embedding = self.embedding_client.embed_text(query)

        if self.debug:
            self.logger.debug(
                f"✅ 向量检索查询向量生成完成，维度: {len(query_embedding)}"
            )

        # 向量检索
        vector_search_body = {
            "knn": {
                "field": vector_field,
                "query_vector": query_embedding,
                "k": top_k,
                "num_candidates": top_k * 4,
                "filter": filter_clause
            },
            "size": top_k,
        }

        if self.debug:
            self.logger.debug(f"🎯 带过滤器的向量检索查询体: {vector_search_body}")

        response = self.client.search(index=self.index_name, body=vector_search_body)
        return self._format_results_with_highlights(response, query, "向量检索")

    def _format_results_with_highlights(
        self, response, query: str = "", search_type: str = ""
    ) -> List[Dict[str, Any]]:
        """格式化检索结果并显示命中词汇"""
        results = []
        total_hits = response["hits"]["total"]

        if self.debug:
            if isinstance(total_hits, dict):
                total_count = total_hits.get("value", 0)
            else:
                total_count = total_hits
            self.logger.debug(
                f"📊 {search_type} 原始检索结果统计 - 总命中数: {total_count}, 返回文档数: {len(response['hits']['hits'])}"
            )

        for i, hit in enumerate(response["hits"]["hits"]):
            # 提取高亮信息
            highlights = hit.get("highlight", {})
            matched_terms = self._extract_matched_terms(highlights, query)
            
            result = {
                "id": hit["_id"],
                "score": hit["_score"],
                "content": hit["_source"].get("content", ""),
                "metadata": hit["_source"].get("metadata", {}),
                "recall_source": "text" if search_type == "文本检索" else "vector",
                "highlights": matched_terms,  # 添加高亮信息到结果中
            }

            # 过滤低分结果
            if result["score"] >= self.similarity_threshold:
                results.append(result)

                if self.debug:
                    # 打印详细的命中文档信息
                    content_preview = (
                        result["content"][:200] + "..."
                        if len(result["content"]) > 200
                        else result["content"]
                    )
                    metadata_info = (
                        ", ".join([f"{k}: {v}" for k, v in result["metadata"].items()])
                        if result["metadata"]
                        else "无元数据"
                    )
                    recall_icon = "📝" if result["recall_source"] == "text" else "🎯"

                    self.logger.debug(
                        f"📄 {recall_icon} {result['recall_source'].upper()}召回文档 #{i+1}:"
                    )
                    self.logger.debug(f"   📋 文档ID: {result['id']}")
                    self.logger.debug(f"   ⭐ 相似度分数: {result['score']:.4f}")

                    # 显示命中的词汇和字段
                    if matched_terms:
                        self.logger.debug("   🎯 命中词汇详情:")
                        for field, terms in matched_terms.items():
                            if terms and isinstance(terms, dict):
                                # 优先显示相关词汇，如果没有则显示所有词汇
                                relevant_terms = terms.get('relevant_terms', [])
                                all_terms = terms.get('all_terms', [])
                                display_terms = relevant_terms if relevant_terms else all_terms
                                if display_terms:
                                    self.logger.debug(
                                        f"      📍 {field}: {', '.join(display_terms)}"
                                    )
                            elif isinstance(terms, list) and terms:
                                # 兼容旧格式（如果是列表）
                                self.logger.debug(
                                    f"      📍 {field}: {', '.join(terms)}"
                                )

                        # 显示高亮片段
                        if highlights:
                            self.logger.debug("   💡 高亮片段:")
                            for field, fragments in highlights.items():
                                if fragments:
                                    for j, fragment in enumerate(
                                        fragments[:2], 1
                                    ):  # 最多显示2个片段
                                        clean_fragment = fragment.replace(
                                            "<mark>", "【"
                                        ).replace("</mark>", "】")
                                        self.logger.debug(
                                            f"      {j}. {field}: {clean_fragment}"
                                        )
                    else:
                        self.logger.debug("   ❓ 未找到具体命中词汇（可能是模糊匹配）")

                    self.logger.debug(f"   📝 内容预览: {content_preview}")
                    self.logger.debug(f"   🏷️  元数据: {metadata_info}")
                    self.logger.debug(f"   📏 内容长度: {len(result['content'])} 字符")
            else:
                if self.debug:
                    self.logger.debug(
                        f"❌ 文档 {hit['_id']} 分数 {hit['_score']:.4f} 低于阈值 {self.similarity_threshold}，已过滤"
                    )

        if self.debug:
            self.logger.debug(
                f"🎯 {search_type} 最终返回 {len(results)} 个有效结果 (查询: '{query}')"
            )
            if results:
                avg_score = sum(r["score"] for r in results) / len(results)
                max_score = max(r["score"] for r in results)
                min_score = min(r["score"] for r in results)
                self.logger.debug(
                    f"📈 分数统计 - 最高: {max_score:.4f}, 最低: {min_score:.4f}, 平均: {avg_score:.4f}"
                )

        return results

    def _extract_matched_terms(self, highlights: dict, query: str) -> dict:
        """从高亮信息中提取匹配的词汇"""
        matched_terms = {}
        
        # 分析查询词
        query_terms = set(query.lower().split())
    
        for field, fragments in highlights.items():
            field_terms = set()
            relevant_terms = set()  # 与查询相关的词汇
    
            for fragment in fragments:
                # 提取被标记的词汇
                import re
    
                marked_words = re.findall(r'<mark>(.*?)</mark>', fragment)
                for word in marked_words:
                    # 清理和标准化词汇
                    clean_word = re.sub(r'[^\w\u4e00-\u9fff]', '', word.lower())
                    if clean_word:
                        field_terms.add(clean_word)
                        # 检查是否与查询词相关（完全匹配或包含关系）
                        if any(clean_word in qterm or qterm in clean_word for qterm in query_terms):
                            relevant_terms.add(clean_word)
    
            if field_terms:
                matched_terms[field] = {
                    'all_terms': list(field_terms),
                    'relevant_terms': list(relevant_terms)
                }
    
        return matched_terms

    def _hybrid_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """混合检索（文本+向量）- 支持RRF和加权融合"""
        if not self.embedding_client:
            raise RuntimeError("混合检索需要embedding客户端")
    
        # 获取向量字段名
        vector_field = self.embedding_config.get("field_name", "embedding")
        
        # 生成查询向量
        if self.debug:
            self.logger.debug(f"🔢 正在生成混合检索查询向量: '{query}'")

        query_embedding = self.embedding_client.embed_text(query)

        if self.debug:
            self.logger.debug(
                f"✅ 混合检索查询向量生成完成，维度: {len(query_embedding)}"
            )
            self.logger.debug(f"🔍 开始执行混合检索 - 融合方法: {self.fusion_method}")

        # 1. 文本检索（带高亮）
        should_queries = []
        highlight_fields = {}

        for field, boost in self.search_fields.items():
            # 对于混合检索，适当降低文本检索的权重
            adjusted_boost = boost * 0.8  # 降低20%权重给向量检索留空间
            should_queries.append(
                {"match": {field: {"query": query, "boost": adjusted_boost, "analyzer": self.search_analyzer}}}
            )

            # 为所有查询字段添加高亮（之前只为部分字段添加）
            if field in ["content", "content_jieba"]:
                highlight_fields[field] = {
                    "fragment_size": 150,
                    "number_of_fragments": 2,
                    "pre_tags": ["<mark>"],
                    "post_tags": ["</mark>"],
                }
            elif "title" in field or "file_path" in field:
                highlight_fields[field] = {
                    "fragment_size": 100,
                    "number_of_fragments": 1,
                    "pre_tags": ["<mark>"],
                    "post_tags": ["</mark>"],
                }
            else:
                # 为其他所有字段也添加高亮设置
                highlight_fields[field] = {
                    "fragment_size": 100,
                    "number_of_fragments": 1,
                    "pre_tags": ["<mark>"],
                    "post_tags": ["</mark>"],
                }

        text_search_body = {
            "query": {"bool": {"should": should_queries}},
            "highlight": {"fields": highlight_fields},
            "size": top_k * 2,
        }

        if self.debug:
            self.logger.debug(f"📝 混合检索-文本部分查询体: {text_search_body}")

        text_response = self.client.search(index=self.index_name, body=text_search_body)

        # 2. 向量检索
        vector_search_body = {
            "knn": {
                "field": vector_field,
                "query_vector": query_embedding,
                "k": top_k * 2,
                "num_candidates": top_k * 4,
            },
            "size": top_k * 2,
        }

        if self.debug:
            self.logger.debug(f"🎯 混合检索-向量部分查询体: {vector_search_body}")

        vector_response = self.client.search(
            index=self.index_name, body=vector_search_body
        )

        # 根据融合方法选择合并策略
        if self.fusion_method == "rrf":
            return self._merge_hybrid_results_with_rrf(
                text_response, vector_response, query, top_k
            )
        else:
            return self._merge_hybrid_results_with_highlights(
                text_response, vector_response, query, top_k
            )

    def _merge_hybrid_results_with_rrf(
        self, text_response, vector_response, query: str, top_k: int
    ) -> List[Dict[str, Any]]:
        """使用RRF算法合并混合检索结果"""
        if self.debug:
            text_hits = len(text_response["hits"]["hits"])
            vector_hits = len(vector_response["hits"]["hits"])
            self.logger.debug(
                f"🔄 开始RRF合并混合检索结果 - 文本检索: {text_hits}个, 向量检索: {vector_hits}个, RRF-K: {self.rrf_k}"
            )

        # 构建文档映射和排名
        text_docs = {}
        vector_docs = {}
        text_ranks = {}
        vector_ranks = {}
        highlights_map = {}

        # 处理文本检索结果
        for rank, hit in enumerate(text_response["hits"]["hits"], 1):
            doc_id = hit["_id"]
            text_docs[doc_id] = {
                "id": doc_id,
                "content": hit["_source"].get("content", ""),
                "metadata": hit["_source"].get("metadata", {}),
                "text_score": hit["_score"],
                "text_rank": rank,
            }
            text_ranks[doc_id] = rank
            highlights_map[doc_id] = hit.get("highlight", {})

            if self.debug:
                matched_terms = self._extract_matched_terms(highlights_map[doc_id], query)
                terms_info = []
                for field, terms in matched_terms.items():
                    if terms and isinstance(terms, dict):
                        relevant_terms = terms.get('relevant_terms', [])
                        all_terms = terms.get('all_terms', [])
                        # 优先显示相关词汇，如果没有则显示所有高亮词汇
                        display_terms = relevant_terms if relevant_terms else all_terms
                        if display_terms:
                            terms_info.append(f"{field}: {', '.join(display_terms)}")
                    elif terms and isinstance(terms, list):
                        terms_info.append(f"{field}: {', '.join(terms)}")
                
                final_terms_info = ', '.join(terms_info) if terms_info else '无'
                # 添加高亮信息调试
                highlight_info = highlights_map[doc_id]
                self.logger.debug(f"🔍 文档 {doc_id} 的高亮信息: {highlight_info}")
                
                matched_terms = self._extract_matched_terms(highlights_map[doc_id], query)
                self.logger.debug(f"🎯 文档 {doc_id} 提取的匹配词: {matched_terms}")
                
                terms_info = []
                for field, terms in matched_terms.items():
                    if terms and isinstance(terms, dict):
                        relevant_terms = terms.get('relevant_terms', [])
                        all_terms = terms.get('all_terms', [])
                        # 优先显示相关词汇，如果没有则显示所有高亮词汇
                        display_terms = relevant_terms if relevant_terms else all_terms
                        self.logger.debug(f"📋 字段 {field}: relevant_terms={relevant_terms}, all_terms={all_terms}, display_terms={display_terms}")
                        if display_terms:
                            terms_info.append(f"{field}: {', '.join(display_terms)}")
                    elif terms and isinstance(terms, list):
                        terms_info.append(f"{field}: {', '.join(terms)}")
                
                final_terms_info = ', '.join(terms_info) if terms_info else '无'
                self.logger.debug(
                    f"📝 文本召回文档: {doc_id}, 排名: {rank}, 分数: {hit['_score']:.4f}, 命中词: {final_terms_info}"
                )

        # 处理向量检索结果
        for rank, hit in enumerate(vector_response["hits"]["hits"], 1):
            doc_id = hit["_id"]
            vector_docs[doc_id] = {
                "id": doc_id,
                "content": hit["_source"].get("content", ""),
                "metadata": hit["_source"].get("metadata", {}),
                "vector_score": hit["_score"],
                "vector_rank": rank,
            }
            vector_ranks[doc_id] = rank

            if self.debug:
                self.logger.debug(
                    f"🎯 向量召回文档: {doc_id}, 排名: {rank}, 分数: {hit['_score']:.4f}"
                )

        # 计算RRF分数
        all_doc_ids = set(text_ranks.keys()) | set(vector_ranks.keys())
        rrf_results = []

        for doc_id in all_doc_ids:
            text_rank = text_ranks.get(doc_id, float('inf'))
            vector_rank = vector_ranks.get(doc_id, float('inf'))
            
            # 计算RRF分数
            rrf_score = 0
            if text_rank != float('inf'):
                rrf_score += 1 / (self.rrf_k + text_rank)
            if vector_rank != float('inf'):
                rrf_score += 1 / (self.rrf_k + vector_rank)
            
            # 确定召回来源
            if text_rank != float('inf') and vector_rank != float('inf'):
                recall_source = "hybrid"
                doc_info = {**text_docs[doc_id], **vector_docs[doc_id]}
            elif text_rank != float('inf'):
                recall_source = "text"
                doc_info = {**text_docs[doc_id], "vector_score": 0.0, "vector_rank": None}
            else:
                recall_source = "vector"
                doc_info = {**vector_docs[doc_id], "text_score": 0.0, "text_rank": None}
                highlights_map[doc_id] = {}  # 向量检索没有高亮
            
            result = {
                "id": doc_id,
                "score": rrf_score,  # 使用RRF分数作为最终分数
                "content": doc_info["content"],
                "metadata": doc_info["metadata"],
                "recall_source": recall_source,
                "text_score": doc_info.get("text_score", 0.0),
                "vector_score": doc_info.get("vector_score", 0.0),
                "text_rank": doc_info.get("text_rank"),
                "vector_rank": doc_info.get("vector_rank"),
                "rrf_score": rrf_score,
                "highlights": self._extract_matched_terms(highlights_map.get(doc_id, {}), query),
            }
            
            rrf_results.append(result)

        # 按RRF分数排序
        rrf_results.sort(key=lambda x: x["rrf_score"], reverse=True)
        
        # 取top_k并应用相似度阈值过滤
        final_results = []
        for result in rrf_results[:top_k]:
            # 对于RRF，我们使用更宽松的阈值策略
            # 因为RRF分数的尺度与原始分数不同
            if result["rrf_score"] > 0:  # RRF分数大于0即表示有意义
                final_results.append(result)

        if self.debug:
            self.logger.debug("📊 RRF混合检索结果统计:")
            text_only = sum(1 for r in final_results if r["recall_source"] == "text")
            vector_only = sum(1 for r in final_results if r["recall_source"] == "vector")
            hybrid_both = sum(1 for r in final_results if r["recall_source"] == "hybrid")

            self.logger.debug(f"   📝 仅文本召回: {text_only}个")
            self.logger.debug(f"   🎯 仅向量召回: {vector_only}个")
            self.logger.debug(f"   🔄 混合召回(文本+向量): {hybrid_both}个")
            self.logger.debug(f"   🎯 最终返回: {len(final_results)}个文档")

            # 详细显示每个文档的RRF信息
            for i, result in enumerate(final_results, 1):
                content_preview = (
                    result["content"][:100] + "..."
                    if len(result["content"]) > 100
                    else result["content"]
                )

                self.logger.debug(f"📄 RRF排名#{i} - 文档ID: {result['id']}")
                
                # 显示RRF计算详情
                rrf_detail = f"RRF分数: {result['rrf_score']:.6f}"
                if result["text_rank"] and result["vector_rank"]:
                    rrf_detail += f" (文本排名: {result['text_rank']}, 向量排名: {result['vector_rank']})"
                elif result["text_rank"]:
                    rrf_detail += f" (仅文本排名: {result['text_rank']})"
                else:
                    rrf_detail += f" (仅向量排名: {result['vector_rank']})"
                
                self.logger.debug(f"   🔢 {rrf_detail}")
                
                # 显示原始分数
                if result["text_score"] > 0 and result["vector_score"] > 0:
                    self.logger.debug(f"   📊 原始分数 - 文本: {result['text_score']:.4f}, 向量: {result['vector_score']:.4f}")
                elif result["text_score"] > 0:
                    self.logger.debug(f"   📊 原始分数 - 文本: {result['text_score']:.4f}")
                else:
                    self.logger.debug(f"   📊 原始分数 - 向量: {result['vector_score']:.4f}")
                
                # 显示命中词汇（仅文本召回有）
                if result["recall_source"] in ["text", "hybrid"] and result["highlights"]:
                    terms_info = []
                    for field, terms in result["highlights"].items():
                        if terms and isinstance(terms, dict):
                            relevant_terms = terms.get('relevant_terms', [])
                            if relevant_terms:  # 只有当有相关词汇时才添加
                                terms_info.append(f"{field}: {', '.join(relevant_terms)}")
                        elif terms and isinstance(terms, list):
                            terms_info.append(f"{field}: {', '.join(terms)}")
                    
                    if terms_info:
                        self.logger.debug(f"   🎯 命中词汇: {', '.join(terms_info)}")
                
                self.logger.debug(f"   🏷️  召回方式: {result['recall_source']}")
                self.logger.debug(f"   📝 内容: {content_preview}")

        return final_results

    def _merge_hybrid_results_with_highlights(
        self, text_response, vector_response, query: str, top_k: int
    ) -> List[Dict[str, Any]]:
        """合并混合检索结果并标记来源（包含高亮信息）- 加权融合方法"""
        results_map = {}
        highlights_map = {}  # 存储高亮信息

        if self.debug:
            text_hits = len(text_response["hits"]["hits"])
            vector_hits = len(vector_response["hits"]["hits"])
            self.logger.debug(
                f"🔄 开始加权合并混合检索结果 - 文本检索: {text_hits}个, 向量检索: {vector_hits}个"
            )

        # 处理文本检索结果
        for hit in text_response["hits"]["hits"]:
            doc_id = hit["_id"]
            score = hit["_score"]
            highlights = hit.get("highlight", {})

            result = {
                "id": doc_id,
                "score": score,
                "content": hit["_source"].get("content", ""),
                "metadata": hit["_source"].get("metadata", {}),
                "recall_source": "text",
                "text_score": score,
                "vector_score": 0.0,
                "hybrid_score": score * self.text_weight,
            }

            results_map[doc_id] = result
            highlights_map[doc_id] = highlights

            if self.debug:
                matched_terms = self._extract_matched_terms(highlights, query)
                terms_info = ", ".join([
                    f"{field}: {', '.join(terms['relevant_terms']) if isinstance(terms, dict) else str(terms)}"
                    for field, terms in matched_terms.items()
                    if terms
                ])
                self.logger.debug(
                    f"📝 文本召回文档: {doc_id}, 分数: {score:.4f}, 命中词: {terms_info or '无'}"
                )

        # 处理向量检索结果
        for hit in vector_response["hits"]["hits"]:
            doc_id = hit["_id"]
            score = hit["_score"]

            if doc_id in results_map:
                # 文档同时被文本和向量召回
                results_map[doc_id]["recall_source"] = "hybrid"
                results_map[doc_id]["vector_score"] = score
                results_map[doc_id]["hybrid_score"] = (
                    results_map[doc_id]["text_score"] * self.text_weight + score * self.vector_weight
                )

                if self.debug:
                    highlights = highlights_map.get(doc_id, {})
                    matched_terms = self._extract_matched_terms(highlights, query)
                    terms_info = ", ".join([
                        f"{field}: {', '.join(terms['relevant_terms']) if isinstance(terms, dict) else str(terms)}"
                        for field, terms in matched_terms.items()
                        if terms
                    ])
                    self.logger.debug(
                        f"🎯 混合召回文档: {doc_id}, 文本分数: {results_map[doc_id]['text_score']:.4f}, 向量分数: {score:.4f}, 混合分数: {results_map[doc_id]['hybrid_score']:.4f}, 命中词: {terms_info or '无'}"
                    )
            else:
                # 仅被向量召回
                result = {
                    "id": doc_id,
                    "score": score,
                    "content": hit["_source"].get("content", ""),
                    "metadata": hit["_source"].get("metadata", {}),
                    "recall_source": "vector",
                    "text_score": 0.0,
                    "vector_score": score,
                    "hybrid_score": score * self.vector_weight,
                }

                results_map[doc_id] = result
                highlights_map[doc_id] = {}

                if self.debug:
                    self.logger.debug(f"🎯 向量召回文档: {doc_id}, 分数: {score:.4f}")

        # 先进行相似度阈值过滤，再按混合分数排序并取top_k
        filtered_results = []
        for result in results_map.values():
            result["score"] = result["hybrid_score"]
            # 对于混合检索，使用更宽松的阈值策略
            threshold = self.similarity_threshold * 0.8  # 降低20%阈值
            if result["score"] >= threshold:
                filtered_results.append(result)

        # 按混合分数排序并取top_k
        filtered_results.sort(key=lambda x: x["hybrid_score"], reverse=True)
        final_results = filtered_results[:top_k]

        if self.debug:
            self.logger.debug("📊 加权混合检索结果统计:")
            text_only = sum(1 for r in final_results if r["recall_source"] == "text")
            vector_only = sum(
                1 for r in final_results if r["recall_source"] == "vector"
            )
            hybrid_both = sum(
                1 for r in final_results if r["recall_source"] == "hybrid"
            )

            self.logger.debug(f"   📝 仅文本召回: {text_only}个")
            self.logger.debug(f"   🎯 仅向量召回: {vector_only}个")
            self.logger.debug(f"   🔄 混合召回(文本+向量): {hybrid_both}个")
            self.logger.debug(f"   🎯 最终返回: {len(final_results)}个文档")

            # 详细显示每个文档的召回信息
            for i, result in enumerate(final_results, 1):
                content_preview = (
                    result["content"][:100] + "..."
                    if len(result["content"]) > 100
                    else result["content"]
                )

                self.logger.debug(f"📄 排名#{i} - 文档ID: {result['id']}")

                # 显示文本召回的命中词汇
                if result["recall_source"] in ["text", "hybrid"]:
                    highlights = highlights_map.get(result["id"], {})
                    matched_terms = self._extract_matched_terms(highlights, query)
                    if matched_terms:
                        terms_info = ", ".join([
                            f"{field}: {', '.join(terms['relevant_terms']) if isinstance(terms, dict) else str(terms)}"
                            for field, terms in matched_terms.items()
                            if terms
                        ])
                        if terms_info:
                            self.logger.debug(f"   🎯 命中词汇: {terms_info}")

                source_info = f"召回方式: {result['recall_source']}"
                if result["recall_source"] == "hybrid":
                    source_info += f" (文本: {result['text_score']:.4f}, 向量: {result['vector_score']:.4f})"
                elif result["recall_source"] == "text":
                    source_info += f" (文本分数: {result['text_score']:.4f})"
                else:
                    source_info += f" (向量分数: {result['vector_score']:.4f})"

                self.logger.debug(f"   🏷️  {source_info}")
                self.logger.debug(f"   ⭐ 最终分数: {result['score']:.4f}")
                self.logger.debug(f"   📝 内容: {content_preview}")

        return final_results
