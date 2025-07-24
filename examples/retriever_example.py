#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Retriever组件使用示例

本示例展示如何使用RetrieverComponent进行文档检索：
1. 单独测试ES检索器
2. 单独测试BM25检索器
3. 构建检索器流水线
"""

import os
import sys

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.components.retriever.bm25_retriever import BM25RetrieverComponent
from rag.components.retriever.es_retriever import ESRetrieverComponent
from rag.components.retriever.retriever_factory import RetrieverFactory
from utils.logger import get_logger, setup_logging

setup_logging(level="DEBUG")


def test_es_retriever():
    """测试ES检索器"""
    logger = get_logger(__name__)
    logger.info("=== 测试ES检索器 ===")

    try:
        # 创建ES检索器
        es_retriever = ESRetrieverComponent(
            name="test_es_retriever",
            config={
                "host": "localhost",
                "port": 9200,
                "username": "elastic",
                "password": "your_password",
                "use_ssl": True,
                "verify_certs": False,
                "index_name": "performance-rag-index",
                "search_type": "text",
                "top_k": 5,
                "debug": True,
            },
        )

        # 初始化
        es_retriever.initialize()

        # 测试检索
        test_queries = ["性能分析工具", "CPU性能优化", "内存泄漏检测"]

        for query in test_queries:
            logger.info(f"\n检索查询: '{query}'")

            # 使用process方法
            result = es_retriever.process({"query": query, "top_k": 3})

            logger.info(f"检索到 {result['result_count']} 个结果")
            for i, doc in enumerate(result["results"], 1):
                logger.info(
                    f"  结果 {i}: {doc['metadata'].get('title', 'N/A')} (得分: {doc['score']:.4f})"
                )

        logger.info("ES检索器测试完成")

    except Exception as e:
        logger.error(f"ES检索器测试失败: {e}")


def test_bm25_retriever():
    """测试BM25检索器"""
    logger = get_logger(__name__)
    logger.info("=== 测试BM25检索器 ===")

    try:
        # 创建BM25检索器
        bm25_retriever = BM25RetrieverComponent(
            name="test_bm25_retriever",
            config={
                "index_name": "test_documents",
                "k1": 1.5,
                "b": 0.75,
                "top_k": 5,
                "storage_path": "./data/bm25_index",
                "auto_load": True,
                "debug": True,
            },
        )

        # 初始化
        bm25_retriever.initialize()

        # 测试检索
        test_queries = ["性能分析工具", "perf命令使用", "CPU性能监控"]

        for query in test_queries:
            logger.info(f"\n检索查询: '{query}'")

            # 使用process方法
            result = bm25_retriever.process({"query": query, "top_k": 3})

            logger.info(f"检索到 {result['result_count']} 个结果")
            for i, doc in enumerate(result["results"], 1):
                logger.info(
                    f"  结果 {i}: {doc['metadata'].get('title', 'N/A')} (得分: {doc['score']:.4f})"
                )

        logger.info("BM25检索器测试完成")

    except Exception as e:
        logger.error(f"BM25检索器测试失败: {e}")


def test_retriever_pipeline():
    """测试检索器流水线"""
    logger = get_logger(__name__)
    logger.info("=== 测试检索器流水线 ===")

    try:
        # 流水线配置
        pipeline_config = {
            "retrievers": [
                {
                    "type": "bm25",
                    "name": "bm25_retriever",
                    "index_name": "test_documents",
                    "k1": 1.5,
                    "b": 0.75,
                    "top_k": 3,
                    "storage_path": "./data/bm25_index",
                    "debug": True,
                }
            ]
        }

        # 创建检索器流水线
        retrievers = RetrieverFactory.create_retriever_pipeline(pipeline_config)

        # 初始化所有检索器
        for retriever in retrievers:
            retriever.initialize()

        # 测试查询
        query = "性能分析工具"
        logger.info(f"\n流水线检索查询: '{query}'")

        all_results = []
        for retriever in retrievers:
            result = retriever.process({"query": query, "top_k": 2})
            all_results.extend(result["results"])
            logger.info(f"{retriever.name} 检索到 {result['result_count']} 个结果")

        logger.info(f"\n流水线总共检索到 {len(all_results)} 个结果")

        logger.info("检索器流水线测试完成")

    except Exception as e:
        logger.error(f"检索器流水线测试失败: {e}")


def main():
    """主函数"""
    logger = get_logger(__name__)
    logger.info("开始Retriever组件示例")

    # 测试BM25检索器（不依赖ES）
    test_bm25_retriever()

    # 测试ES检索器（需要ES服务运行）
    # test_es_retriever()

    # 测试检索器流水线
    test_retriever_pipeline()

    logger.info("Retriever组件示例完成")


if __name__ == "__main__":
    main()
