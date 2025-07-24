#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Elasticsearch 向量索引器使用示例

本示例展示如何使用 OpenAI embedding 接口对文档进行向量化，
然后存储到 Elasticsearch 并进行向量检索：
1. 加载测试数据文件
2. 使用 OpenAI 接口生成文档向量
3. 创建支持向量搜索的 ES 索引
4. 索引文档和向量
5. 执行文本搜索和向量搜索
6. 混合检索（文本 + 向量）
"""

import os
import sys
import time

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.components.index.es_indexer import ESIndexerComponent
from rag.components.embedding.openai_embedding import OpenAIEmbedding
from utils.logger import get_logger, setup_logging

setup_logging(level="DEBUG")


def load_test_data():
    """加载测试数据文件"""
    test_data_dir = "/Users/caixiaomeng/Projects/Python/PerformanceRag/test_cases/test_data"
    documents = []
    
    # 加载混合语言数据
    with open(os.path.join(test_data_dir, "mixed_language_data.txt"), "r", encoding="utf-8") as f:
        content = f.read()
        documents.append({
            "id": "mixed_lang_doc",
            "content": content,
            "metadata": {
                "title": "Multi-language Performance Analysis Guide",
                "category": "Performance Analysis",
                "language": "mixed",
                "tags": ["performance", "analysis", "multilingual"]
            }
        })
    
    # 加载语义分割数据
    with open(os.path.join(test_data_dir, "semantic_splitter_data.txt"), "r", encoding="utf-8") as f:
        content = f.read()
        documents.append({
            "id": "ai_development_doc",
            "content": content,
            "metadata": {
                "title": "人工智能在软件开发中的应用",
                "category": "AI Development",
                "language": "chinese",
                "tags": ["AI", "machine learning", "software development"]
            }
        })
    
    # 加载文本分割数据
    with open(os.path.join(test_data_dir, "text_splitter_data.txt"), "r", encoding="utf-8") as f:
        content = f.read()
        documents.append({
            "id": "perf_tool_doc",
            "content": content,
            "metadata": {
                "title": "Linux性能分析工具Perf简介",
                "category": "Linux Tools",
                "language": "chinese",
                "tags": ["linux", "perf", "performance", "tools"]
            }
        })
    
    # 加载递归分割数据
    with open(os.path.join(test_data_dir, "recursive_splitter_data.md"), "r", encoding="utf-8") as f:
        content = f.read()
        documents.append({
            "id": "perf_guide_doc",
            "content": content,
            "metadata": {
                "title": "Linux性能分析工具Perf完整指南",
                "category": "Linux Tools",
                "language": "chinese",
                "format": "markdown",
                "tags": ["linux", "perf", "guide", "tutorial"]
            }
        })
    
    return documents


def create_vector_mapping():
    """创建支持向量搜索的映射配置"""
    return {
        "properties": {
            "content": {
                "type": "text",
                "analyzer": "standard",
                "fields": {
                    "keyword": {
                        "type": "keyword",
                        "ignore_above": 256
                    }
                }
            },
            "content_vector": {
                "type": "dense_vector",
                "dims": 2048,  # 修改为2048维，匹配实际向量维度
                "index": True,
                "similarity": "cosine"
            },
            "metadata": {
                "type": "object",
                "properties": {
                    "title": {"type": "text", "analyzer": "standard"},
                    "category": {"type": "keyword"},
                    "language": {"type": "keyword"},
                    "tags": {"type": "keyword"},
                    "format": {"type": "keyword"}
                }
            },
            "timestamp": {"type": "date"}
        }
    }


def main():
    """主函数"""
    logger = get_logger(__name__)
    logger.info("开始 Elasticsearch 向量索引器示例")
    
    try:
        # 1. 初始化 OpenAI Embedding 客户端
        logger.info("初始化 OpenAI Embedding 客户端...")
        embedding_client = OpenAIEmbedding(
            model="embedding-3",
            api_key="9dec52aca0e144fc98f3ab8d407e9a57.MNrL3f2b8cgfRsHh",
            api_base="https://open.bigmodel.cn/api/paas/v4",
            batch_size=10
        )
        
        # 2. 创建支持向量搜索的 ES 索引器
        logger.info("创建 Elasticsearch 向量索引器...")
        es_indexer = ESIndexerComponent(
            name="vector_es_indexer",
            config={
                "index_name": "vector_documents",
                "host": "localhost",
                "port": 9200,
                "username": "elastic",
                "password": "sPxLec=NGSFmUT_7+74R",
                "use_ssl": True,
                "verify_certs": False,
                # 支持向量搜索的映射配置
                "mapping": create_vector_mapping(),
                "settings": {
                    "number_of_shards": 1,
                    "number_of_replicas": 0,
                    "analysis": {
                        "analyzer": {
                            "default": {
                                "type": "standard"
                            }
                        }
                    }
                }
            }
        )
        
        # 3. 初始化索引器
        logger.info("初始化索引器...")
        es_indexer.initialize()
        
        # 4. 检查并创建索引
        logger.info("检查索引状态...")
        if es_indexer.client.indices.exists(index=es_indexer.index_name):
            logger.warning(f"索引 {es_indexer.index_name} 已存在")
            logger.info("如果需要重新创建索引，请手动删除后重新运行")
            logger.info(f"删除命令: curl -X DELETE 'https://localhost:9200/{es_indexer.index_name}' -u elastic:sPxLec=NGSFmUT_7+74R -k")
        
        if es_indexer.create_index():
            logger.info("向量索引准备就绪")
        else:
            logger.error("索引创建失败")
            return
        
        # 5. 加载测试数据
        logger.info("加载测试数据...")
        documents = load_test_data()
        logger.info(f"加载了 {len(documents)} 个文档")
        
        # 6. 生成文档向量
        logger.info("生成文档向量...")
        texts_to_embed = [doc["content"] for doc in documents]
        
        try:
            vectors = embedding_client.embed_texts(texts_to_embed)
            logger.info(f"成功生成 {len(vectors)} 个向量，每个向量维度: {len(vectors[0])}")
        except Exception as e:
            logger.error(f"向量生成失败: {e}")
            return
        
        # 7. 为文档添加向量
        for i, doc in enumerate(documents):
            doc["content_vector"] = vectors[i]
            doc["timestamp"] = "2024-01-01T00:00:00Z"
        
        # 8. 索引文档
        logger.info("开始索引文档和向量...")
        try:
            # 准备批量操作
            actions = []
            for i, doc in enumerate(documents):
                doc_id = doc.get("id", f"doc_{i}")
                action = {"_index": es_indexer.index_name, "_id": doc_id, "_source": doc}
                actions.append(action)
            
            # 批量索引
            from elasticsearch.helpers import bulk
            success_count, failed_items = bulk(
                es_indexer.client, actions, chunk_size=100, request_timeout=60
            )
            
            if failed_items:
                logger.error(f"索引失败的文档数量: {len(failed_items)}")
                for item in failed_items:
                    logger.error(f"失败详情: {item}")
                return
            else:
                logger.info(f"成功索引 {success_count} 个文档和向量")
                
        except Exception as e:
            logger.error(f"文档索引失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return
        
        # 等待索引刷新
        logger.info("等待索引刷新...")
        time.sleep(3)
        
        # 9. 执行文本搜索测试
        logger.info("\n=== 文本搜索测试 ===")
        
        text_queries = [
            "性能分析工具",
            "人工智能代码生成",
            "Linux perf 命令",
            "machine learning"
        ]
        
        for query in text_queries:
            logger.info(f"\n文本搜索查询: '{query}'")
            results = es_indexer.search(query, top_k=3)
            
            if results:
                for i, result in enumerate(results, 1):
                    logger.info(f"  结果 {i}: {result['metadata']['title']} (得分: {result['score']:.4f})")
            else:
                logger.info("  未找到相关结果")
        
        # 10. 执行向量搜索测试
        logger.info("\n=== 向量搜索测试 ===")
        
        vector_queries = [
            "如何使用性能分析工具优化程序",
            "AI在软件开发中的应用场景",
            "Linux系统性能监控方法"
        ]
        
        for query in vector_queries:
            logger.info(f"\n向量搜索查询: '{query}'")
            
            # 生成查询向量
            try:
                query_vector = embedding_client.embed_text(query)
                
                # 构建向量搜索查询
                vector_search_body = {
                    "query": {
                        "script_score": {
                            "query": {"match_all": {}},
                            "script": {
                                "source": "cosineSimilarity(params.query_vector, 'content_vector') + 1.0",
                                "params": {"query_vector": query_vector}
                            }
                        }
                    },
                    "size": 3
                }
                
                # 执行向量搜索
                response = es_indexer.client.search(index=es_indexer.index_name, body=vector_search_body)
                
                # 处理结果
                for i, hit in enumerate(response["hits"]["hits"], 1):
                    score = hit["_score"]
                    title = hit["_source"]["metadata"]["title"]
                    logger.info(f"  结果 {i}: {title} (向量得分: {score:.4f})")
                    
            except Exception as e:
                logger.error(f"向量搜索失败: {e}")
        
        # 11. 混合搜索测试（文本 + 向量）
        logger.info("\n=== 混合搜索测试 ===")
        
        hybrid_query = "Linux性能优化技巧"
        logger.info(f"\n混合搜索查询: '{hybrid_query}'")
        
        try:
            # 生成查询向量
            query_vector = embedding_client.embed_text(hybrid_query)
            
            # 构建混合搜索查询
            hybrid_search_body = {
                "query": {
                    "bool": {
                        "should": [
                            {
                                "multi_match": {
                                    "query": hybrid_query,
                                    "fields": ["content^1.5", "metadata.title^2", "metadata.tags"],
                                    "boost": 1.0
                                }
                            },
                            {
                                "script_score": {
                                    "query": {"match_all": {}},
                                    "script": {
                                        "source": "cosineSimilarity(params.query_vector, 'content_vector')",
                                        "params": {"query_vector": query_vector}
                                    },
                                    "boost": 2.0
                                }
                            }
                        ]
                    }
                },
                "size": 3
            }
            
            # 执行混合搜索
            response = es_indexer.client.search(index=es_indexer.index_name, body=hybrid_search_body)
            
            # 处理结果
            for i, hit in enumerate(response["hits"]["hits"], 1):
                score = hit["_score"]
                title = hit["_source"]["metadata"]["title"]
                logger.info(f"  结果 {i}: {title} (混合得分: {score:.4f})")
                
        except Exception as e:
            logger.error(f"混合搜索失败: {e}")
        
        # 12. 索引统计信息
        logger.info("\n=== 索引统计 ===")
        try:
            stats = es_indexer.client.indices.stats(index=es_indexer.index_name)
            doc_count = stats['indices'][es_indexer.index_name]['total']['docs']['count']
            store_size = stats['indices'][es_indexer.index_name]['total']['store']['size_in_bytes']
            
            logger.info(f"索引名称: {es_indexer.index_name}")
            logger.info(f"文档数量: {doc_count}")
            logger.info(f"存储大小: {store_size} 字节")
            logger.info("向量维度: 2048")  # 修改为正确的维度
            
        except Exception as e:
            logger.warning(f"获取索引统计失败: {e}")
        
        logger.info("\nElasticsearch 向量索引器示例完成!")
        logger.info("\n功能特性:")
        logger.info("- ✅ OpenAI embedding 接口集成")
        logger.info("- ✅ 文档向量化和存储")
        logger.info("- ✅ 文本搜索")
        logger.info("- ✅ 向量搜索")
        logger.info("- ✅ 混合搜索（文本 + 向量）")
        logger.info("- ✅ 余弦相似度计算")
        
    except Exception as e:
        logger.error(f"示例执行失败: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()