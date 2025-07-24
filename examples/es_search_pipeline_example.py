#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ES搜索Pipeline使用示例

本示例展示如何使用ES搜索Pipeline：
1. 执行检索
2. 重排序结果
3. 生成回答
"""

import os
import sys

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.pipeline.factory import clear_cache, create_pipeline
from utils.logger import get_logger, setup_logging

setup_logging(level="DEBUG")


def test_es_search_pipeline():
    """测试ES搜索Pipeline"""
    logger = get_logger(__name__)
    logger.info("=== 测试ES搜索Pipeline ===")

    try:
        # 创建ES搜索Pipeline
        logger.info("创建ES搜索Pipeline...")
        pipeline = create_pipeline("es_search_pipeline")
        logger.info(f"Pipeline创建完成，组件列表: {pipeline.list_components()}")

        # 测试查询列表
        test_queries = [
            "什么是perf工具？",
            "如何进行性能分析？",
            "Linux性能监控有哪些方法？",
            "人工智能在软件开发中的应用",
            "机器学习算法有哪些？",
        ]

        for i, query in enumerate(test_queries, 1):
            logger.info(f"\n--- 测试查询 {i}: {query} ---")

            # 执行搜索流程
            result = pipeline.run(
                {"query": query, "top_k": 3}, entry_point="es_retriever"
            )

            # 显示结果
            if result.get("success", False):
                logger.info(f"检索到 {result.get('result_count', 0)} 个相关文档")

                # 显示检索结果
                if "retrieved_docs" in result:
                    for j, doc in enumerate(result["retrieved_docs"][:3], 1):
                        logger.info(
                            f"文档 {j}: {doc.get('metadata', {}).get('title', 'Unknown')}"
                        )
                        logger.info(f"相似度: {doc.get('score', 0):.4f}")

                # 显示生成的回答
                if "generated_answer" in result:
                    logger.info(f"生成的回答: {result['generated_answer'][:200]}...")

            else:
                logger.error(f"搜索失败: {result.get('error', '未知错误')}")

        logger.info("ES搜索Pipeline测试完成")

    except Exception as e:
        logger.error(f"ES搜索Pipeline测试失败: {e}")
        import traceback

        traceback.print_exc()


def test_different_search_types():
    """测试不同的搜索类型"""
    logger = get_logger(__name__)
    logger.info("=== 测试不同搜索类型 ===")

    search_types = ["text", "vector", "hybrid"]
    query = "什么是perf工具？"

    for search_type in search_types:
        logger.info(f"\n--- 测试 {search_type} 搜索 ---")

        try:
            # 动态修改搜索类型
            pipeline = create_pipeline("es_search_pipeline")

            # 执行搜索
            result = pipeline.run(
                {"query": query, "search_type": search_type, "top_k": 3},
                entry_point="es_retriever",
            )

            if result.get("success", False):
                logger.info(
                    f"{search_type} 搜索成功，找到 {result.get('result_count', 0)} 个结果"
                )
            else:
                logger.error(
                    f"{search_type} 搜索失败: {result.get('error', '未知错误')}"
                )

        except Exception as e:
            logger.error(f"{search_type} 搜索异常: {e}")


def interactive_search():
    """交互式搜索"""
    logger = get_logger(__name__)
    logger.info("=== 交互式搜索模式 ===")
    logger.info("输入查询问题，输入 'quit' 退出")

    try:
        pipeline = create_pipeline("es_search_pipeline")

        while True:
            query = input("\n请输入查询: ").strip()

            if query.lower() in ["quit", "exit", "q"]:
                break

            if not query:
                continue

            logger.info(f"搜索: {query}")

            result = pipeline.run(
                {"query": query, "top_k": 3}, entry_point="es_retriever"
            )

            if result.get("success", False):
                print(f"\n找到 {result.get('result_count', 0)} 个相关结果:")

                if "retrieved_docs" in result:
                    for i, doc in enumerate(result["retrieved_docs"][:3], 1):
                        print(f"{i}. {doc.get('metadata', {}).get('title', 'Unknown')}")
                        print(f"   相似度: {doc.get('score', 0):.4f}")
                        print(f"   内容: {doc.get('content', '')[:100]}...")

                if "generated_answer" in result:
                    print(f"\n回答: {result['generated_answer']}")
            else:
                print(f"搜索失败: {result.get('error', '未知错误')}")

    except KeyboardInterrupt:
        logger.info("用户中断搜索")
    except Exception as e:
        logger.error(f"交互式搜索异常: {e}")


def main():
    """主函数"""
    logger = get_logger(__name__)
    logger.info("开始ES搜索Pipeline示例")

    # 基本搜索测试
    test_es_search_pipeline()

    print("\n" + "=" * 50 + "\n")

    # 不同搜索类型测试
    test_different_search_types()

    print("\n" + "=" * 50 + "\n")

    # 交互式搜索（可选）
    try:
        interactive_search()
    except Exception as e:
        logger.info(f"跳过交互式搜索: {e}")

    # 清理缓存
    clear_cache()
    logger.info("缓存已清理")

    logger.info("ES搜索Pipeline示例完成")


if __name__ == "__main__":
    main()
