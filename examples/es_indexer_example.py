#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Elasticsearch索引器使用示例

本示例展示如何使用ESIndexerComponent进行文档索引和搜索：
1. 连接Elasticsearch服务
2. 创建索引
3. 索引文档
4. 执行搜索
5. 获取特定文档
6. 管理索引
"""

import os
import sys
import time

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.components.index.es_indexer import ESIndexerComponent
from utils.logger import get_logger, setup_logging

setup_logging(level="DEBUG")


def load_test_data():
    """加载测试数据文件"""
    test_data_dir = (
        "/Users/caixiaomeng/Projects/Python/PerformanceRag/test_cases/test_data"
    )
    documents = []

    # 加载混合语言数据
    with open(
        os.path.join(test_data_dir, "mixed_language_data.txt"), "r", encoding="utf-8"
    ) as f:
        content = f.read()
        documents.append(
            {
                "id": "mixed_lang_doc",
                "content": content,
                "metadata": {
                    "title": "Multi-language Performance Analysis Guide",
                    "category": "Performance Analysis",
                    "language": "mixed",
                    "tags": ["performance", "analysis", "multilingual"],
                    "author": "System",
                    "created_at": "2024-01-01T00:00:00Z",
                },
                "timestamp": "2024-01-01T00:00:00Z",
            }
        )

    # 加载语义分割数据
    with open(
        os.path.join(test_data_dir, "semantic_splitter_data.txt"), "r", encoding="utf-8"
    ) as f:
        content = f.read()
        documents.append(
            {
                "id": "ai_development_doc",
                "content": content,
                "metadata": {
                    "title": "人工智能在软件开发中的应用",
                    "category": "AI Development",
                    "language": "chinese",
                    "tags": ["AI", "machine learning", "software development"],
                    "author": "AI Expert",
                    "created_at": "2024-01-02T00:00:00Z",
                },
                "timestamp": "2024-01-02T00:00:00Z",
            }
        )

    # 加载文本分割数据
    with open(
        os.path.join(test_data_dir, "text_splitter_data.txt"), "r", encoding="utf-8"
    ) as f:
        content = f.read()
        documents.append(
            {
                "id": "perf_tool_doc",
                "content": content,
                "metadata": {
                    "title": "Linux性能分析工具Perf简介",
                    "category": "Linux Tools",
                    "language": "chinese",
                    "tags": ["linux", "perf", "performance", "tools"],
                    "author": "Linux Expert",
                    "created_at": "2024-01-03T00:00:00Z",
                },
                "timestamp": "2024-01-03T00:00:00Z",
            }
        )

    # 加载递归分割数据
    with open(
        os.path.join(test_data_dir, "recursive_splitter_data.md"), "r", encoding="utf-8"
    ) as f:
        content = f.read()
        documents.append(
            {
                "id": "perf_guide_doc",
                "content": content,
                "metadata": {
                    "title": "Linux性能分析工具Perf完整指南",
                    "category": "Linux Tools",
                    "language": "chinese",
                    "format": "markdown",
                    "tags": ["linux", "perf", "guide", "tutorial"],
                    "author": "Technical Writer",
                    "created_at": "2024-01-04T00:00:00Z",
                },
                "timestamp": "2024-01-04T00:00:00Z",
            }
        )

    return documents


def check_elasticsearch_connection(
    host="localhost", port=9200, username=None, password=None, use_ssl=True
):
    """检查Elasticsearch连接"""
    try:
        from elasticsearch import Elasticsearch

        # 对于 ES 8.x，通过 URL scheme 指定协议
        if use_ssl:
            hosts = [f"https://{host}:{port}"]
        else:
            hosts = [f"http://{host}:{port}"]

        es_config = {
            "hosts": hosts,
            "verify_certs": False,  # 开发环境可以设为 False
        }

        if username and password:
            es_config["basic_auth"] = (username, password)

        # 移除 use_ssl 参数
        # if use_ssl:
        #     es_config["use_ssl"] = True

        client = Elasticsearch(**es_config)
        return client.ping()

    except Exception as e:
        print(f"连接检查失败: {e}")
        return False


def main():
    """主函数"""
    logger = get_logger(__name__)
    logger.info("开始Elasticsearch索引器示例")

    try:
        # 0. 检查Elasticsearch连接
        logger.info("检查Elasticsearch连接...")
        if not check_elasticsearch_connection(
            host="localhost",
            port=9200,
            username="elastic",
            password="sPxLec=NGSFmUT_7+74R",
            use_ssl=True,  # 启用 SSL
        ):
            logger.error("无法连接到Elasticsearch服务，请确保ES服务正在运行")
            return

        logger.info("Elasticsearch连接正常")

        # 1. 创建ES索引器
        logger.info("创建Elasticsearch索引器...")
        es_indexer = ESIndexerComponent(
            name="test_es_indexer",
            config={
                "index_name": "test_documents",
                "host": "localhost",
                "port": 9200,
                "batch_size": 50,
                "enable_debug": True,
                "username": "elastic",
                "password": "sPxLec=NGSFmUT_7+74R",
                "use_ssl": True,  # 启用 SSL
                "verify_certs": False,  # 如果使用自签名证书
                # 自定义映射配置
                "mapping": {
                    "properties": {
                        "content": {
                            "type": "text",
                            "analyzer": "standard",
                            "fields": {
                                "keyword": {"type": "keyword", "ignore_above": 256}
                            },
                        },
                        "metadata": {
                            "type": "object",
                            "properties": {
                                "title": {"type": "text", "analyzer": "standard"},
                                "category": {"type": "keyword"},
                                "language": {"type": "keyword"},
                                "tags": {"type": "keyword"},
                                "author": {"type": "keyword"},
                                "created_at": {"type": "date"},
                            },
                        },
                        "timestamp": {"type": "date"},
                    }
                },
                # 自定义设置配置
                "settings": {
                    "number_of_shards": 1,
                    "number_of_replicas": 0,
                    "analysis": {"analyzer": {"default": {"type": "standard"}}},
                },
            },
        )

        # 2. 初始化索引器
        logger.info("初始化索引器...")
        es_indexer.initialize()

        # 3. 检查索引是否存在，如果需要可以手动删除
        logger.info("检查索引状态...")
        if es_indexer.client.indices.exists(index=es_indexer.index_name):
            logger.warning(f"索引 {es_indexer.index_name} 已存在")
            logger.info("如果需要重新创建索引，请手动删除后重新运行")
            logger.info(
                f"删除命令: curl -X DELETE 'https://localhost:9200/{es_indexer.index_name}' -u elastic:sPxLec=NGSFmUT_7+74R -k"
            )

        # 4. 创建索引（如果不存在）
        logger.info("确保索引存在...")
        if es_indexer.create_index():
            logger.info("索引准备就绪")
        else:
            logger.error("索引创建失败")
            return

        # 5. 加载测试数据
        logger.info("加载测试数据...")
        documents = load_test_data()
        logger.info(f"加载了 {len(documents)} 个文档")

        # 6. 索引文档
        logger.info("开始索引文档...")
        index_result = es_indexer.process({"documents": documents})
        logger.info(f"索引结果: {index_result}")

        # 检查索引是否成功
        if index_result.get("indexed", False):
            logger.info(f"成功索引 {index_result.get('document_count', 0)} 个文档")
        else:
            logger.error("文档索引失败")
            return

        # 等待索引刷新
        logger.info("等待索引刷新...")
        time.sleep(2)

        # 7. 执行搜索测试
        logger.info("\n=== 搜索测试 ===")

        # 测试查询
        queries = [
            "性能分析工具",
            "perf命令使用",
            "人工智能代码生成",
            "machine learning algorithms",
            "CPU cycles performance",
            "缓存未命中",
            "深度学习模型",
            "Linux tools",
            "Elasticsearch",
        ]

        for query in queries:
            logger.info(f"\n搜索查询: '{query}'")
            search_result = es_indexer.process({"query": query, "top_k": 3})

            # 使用正确的字段名
            if search_result.get("results"):
                for i, result in enumerate(search_result["results"], 1):
                    logger.info(
                        f"  结果 {i}: {result['metadata']['title']} (得分: {result['score']:.4f})"
                    )
                    logger.info(
                        f"    类别: {result['metadata'].get('category', 'N/A')}"
                    )
                    logger.info(
                        f"    标签: {', '.join(result['metadata'].get('tags', []))}"
                    )
            else:
                logger.info("  未找到相关结果")

        # 8. 高级搜索测试
        logger.info("\n=== 高级搜索测试 ===")

        # 按类别搜索
        logger.info("\n按类别搜索 'Linux Tools':")
        advanced_search_result = es_indexer.search(query="Linux Tools", top_k=5)

        for i, result in enumerate(advanced_search_result, 1):
            logger.info(
                f"  结果 {i}: {result['metadata']['title']} (得分: {result['score']:.4f})"
            )

        # 9. 获取特定文档
        logger.info("\n=== 文档获取测试 ===")
        doc = es_indexer.get_document("ai_development_doc")

        if doc:
            logger.info(f"获取文档: {doc['metadata']['title']}")
            logger.info(f"内容长度: {len(doc['content'])} 字符")
            logger.info(f"作者: {doc['metadata'].get('author', 'N/A')}")
        else:
            logger.info("未找到指定文档")

        # 10. 索引统计信息
        logger.info("\n=== 索引统计 ===")
        try:
            # 获取索引统计
            stats = es_indexer.client.indices.stats(index=es_indexer.index_name)
            doc_count = stats["indices"][es_indexer.index_name]["total"]["docs"][
                "count"
            ]
            store_size = stats["indices"][es_indexer.index_name]["total"]["store"][
                "size_in_bytes"
            ]

            logger.info(f"索引名称: {es_indexer.index_name}")
            logger.info(f"文档数量: {doc_count}")
            logger.info(f"存储大小: {store_size} 字节")

            # 获取映射信息
            mapping = es_indexer.client.indices.get_mapping(index=es_indexer.index_name)
            logger.info(
                f"映射字段数: {len(mapping[es_indexer.index_name]['mappings']['properties'])}"
            )

        except Exception as e:
            logger.warning(f"获取索引统计失败: {e}")

        # 11. 演示多索引操作（新增功能）
        logger.info("\n=== 多索引操作演示 ===")

        # 创建一个临时索引用于演示
        temp_index_name = "temp_test_index"
        logger.info(f"创建临时索引: {temp_index_name}")

        if es_indexer.create_index(index_name=temp_index_name):
            logger.info(f"临时索引 {temp_index_name} 创建成功")

            # 向临时索引添加一些文档
            temp_docs = documents[:2]  # 只取前两个文档
            if es_indexer.index_documents(temp_docs, index_name=temp_index_name):
                logger.info(f"成功向 {temp_index_name} 索引了 {len(temp_docs)} 个文档")

                # 从临时索引搜索
                temp_results = es_indexer.search(
                    "性能", top_k=2, index_name=temp_index_name
                )
                logger.info(f"从 {temp_index_name} 搜索到 {len(temp_results)} 个结果")

                # 注意：这里不再提供删除索引的功能，需要手动删除
                logger.info(
                    f"演示完成，如需删除临时索引 {temp_index_name}，请手动执行:"
                )
                logger.info(
                    f"curl -X DELETE 'https://localhost:9200/{temp_index_name}' -u elastic:sPxLec=NGSFmUT_7+74R -k"
                )

        logger.info("\nElasticsearch索引器示例完成!")
        logger.info("\n注意事项:")
        logger.info(
            "- 索引删除功能已移除，如需删除索引请使用 Elasticsearch API 或管理工具"
        )
        logger.info("- 所有方法现在都支持可选的 index_name 参数，提供更好的灵活性")
        logger.info("- 建议在生产环境中使用专门的索引管理工具")

    except Exception as e:
        logger.error(f"示例执行失败: {str(e)}")
        import traceback

        logger.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
