#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
完整RAG流水线示例

本示例展示如何使用配置文件构建和运行完整的RAG流水线：
1. 从YAML配置文件加载流水线配置
2. 构建包含loader、splitter、indexer、retriever的完整流水线
3. 执行文档索引流程
4. 执行文档检索流程
"""

import os
import sys
import time

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.pipeline.factory import create_pipeline
from utils.config import config
from utils.logger import get_logger, setup_logging

# 确保所有组件都被导入和注册

setup_logging(level="DEBUG")


def test_bm25_pipeline():
    """测试BM25流水线"""
    logger = get_logger(__name__)
    logger.info("=== 测试BM25流水线 ===")

    try:
        # 使用独立的配置文件
        pipeline = create_pipeline("bm25_pipeline")
        
        # 构建流水线
        pipeline.build()
        logger.info("BM25流水线构建完成")

        # 执行索引流程（只到索引器，不包括检索器）
        logger.info("开始执行文档索引流程...")
        
        # 手动执行索引流程：loader -> splitter -> indexer
        loader = pipeline.components.get("document_loader")
        splitter = pipeline.components.get("text_splitter")
        indexer = pipeline.components.get("bm25_indexer")
        
        if loader and splitter and indexer:
            # 加载文档
            docs_result = loader.process({})
            logger.info(f"加载了 {len(docs_result.get('documents', []))} 个文档")
            
            # 分割文档
            split_result = splitter.process(docs_result)
            logger.info(f"分割得到 {len(split_result.get('documents', []))} 个文档块")
            
            # 索引文档
            index_result = indexer.process(split_result)
            logger.info(f"索引流程完成: {index_result.get('metadata', {})}")
        else:
            logger.error("缺少必要的组件")
            return

        # 等待索引完成
        time.sleep(2)

        # 执行检索流程
        logger.info("开始执行文档检索流程...")
        test_queries = ["性能分析工具", "perf命令使用", "人工智能开发", "机器学习算法"]

        for query in test_queries:
            logger.info(f"\n检索查询: '{query}'")

            # 直接调用检索器组件
            retriever = pipeline.components.get("bm25_retriever")
            if retriever:
                search_result = retriever.process({"query": query, "top_k": 3})

                logger.info(f"检索到 {search_result['result_count']} 个结果")
                for i, doc in enumerate(search_result["results"], 1):
                    title = doc["metadata"].get("title", "N/A")
                    score = doc["score"]
                    logger.info(f"  结果 {i}: {title} (得分: {score:.4f})")
            else:
                logger.warning("未找到检索器组件")

        logger.info("BM25流水线测试完成")

    except Exception as e:
        logger.error(f"BM25流水线测试失败: {e}")
        import traceback
        traceback.print_exc()


def test_es_pipeline():
    """测试ES流水线（需要ES服务运行）"""
    logger = get_logger(__name__)
    logger.info("=== 测试ES流水线 ===")

    try:
        # 使用工厂方法创建流水线
        pipeline = create_pipeline("es_rag_pipeline")
        logger.info("ES流水线构建完成")

        # 执行索引流程
        logger.info("开始执行文档索引流程...")
        index_result = pipeline.run({})
        logger.info(f"索引流程完成: {index_result.get('metadata', {})}")

        # 等待索引完成
        time.sleep(2)

        # 执行检索流程
        logger.info("开始执行文档检索流程...")
        test_queries = ["性能分析工具", "CPU性能优化"]

        for query in test_queries:
            logger.info(f"\n检索查询: '{query}'")

            # 直接调用检索器组件
            retriever = pipeline.components.get("es_retriever")
            if retriever:
                search_result = retriever.process({"query": query, "top_k": 3})

                logger.info(f"检索到 {search_result['result_count']} 个结果")
                for i, doc in enumerate(search_result["results"], 1):
                    title = doc["metadata"].get("title", "N/A")
                    score = doc["score"]
                    logger.info(f"  结果 {i}: {title} (得分: {score:.4f})")
            else:
                logger.warning("未找到检索器组件")

        logger.info("ES流水线测试完成")

    except Exception as e:
        logger.error(f"ES流水线测试失败: {e}")
        logger.info("请确保Elasticsearch服务正在运行")


def test_hybrid_pipeline():
    """测试混合检索流水线"""
    logger = get_logger(__name__)
    logger.info("=== 测试混合检索流水线 ===")

    try:
        # 使用工厂方法创建流水线
        pipeline = create_pipeline("hybrid_rag_pipeline")
        logger.info("混合检索流水线构建完成")

        # 执行索引流程
        logger.info("开始执行文档索引流程...")
        index_result = pipeline.run({})
        logger.info(f"索引流程完成: {index_result.get('metadata', {})}")

        # 等待索引完成
        time.sleep(3)

        # 执行检索流程
        logger.info("开始执行文档检索流程...")
        query = "性能分析工具"
        logger.info(f"\n检索查询: '{query}'")

        # 分别调用BM25和ES检索器
        bm25_retriever = pipeline.components.get("bm25_retriever")
        es_retriever = pipeline.components.get("es_retriever")

        all_results = []

        if bm25_retriever:
            bm25_result = bm25_retriever.process({"query": query, "top_k": 3})
            logger.info(f"BM25检索到 {bm25_result['result_count']} 个结果")
            all_results.extend(bm25_result["results"])

        if es_retriever:
            es_result = es_retriever.process({"query": query, "top_k": 3})
            logger.info(f"ES检索到 {es_result['result_count']} 个结果")
            all_results.extend(es_result["results"])

        # 简单的结果合并（去重）
        unique_results = {}
        for result in all_results:
            doc_id = result.get("id", result.get("content", "")[:50])
            if (
                doc_id not in unique_results
                or result["score"] > unique_results[doc_id]["score"]
            ):
                unique_results[doc_id] = result

        # 按分数排序
        sorted_results = sorted(
            unique_results.values(), key=lambda x: x["score"], reverse=True
        )

        logger.info(f"\n混合检索总共得到 {len(sorted_results)} 个唯一结果")
        for i, doc in enumerate(sorted_results[:5], 1):
            title = doc["metadata"].get("title", "N/A")
            score = doc["score"]
            logger.info(f"  结果 {i}: {title} (得分: {score:.4f})")

        logger.info("混合检索流水线测试完成")

    except Exception as e:
        logger.error(f"混合检索流水线测试失败: {e}")
        import traceback

        traceback.print_exc()


def test_pipeline_components():
    """测试流水线组件状态"""
    logger = get_logger(__name__)
    logger.info("=== 测试流水线组件状态 ===")

    try:
        # 导入组件注册表
        from rag.pipeline.registry import ComponentRegistry

        # 列出所有注册的组件
        all_components = ComponentRegistry.list_components()
        logger.info("已注册的组件类型:")

        for comp_type, components in all_components.items():
            logger.info(f"  {comp_type}: {list(components.keys())}")

        # 测试配置加载
        logger.info("\n测试配置加载...")
        if hasattr(config, "full_rag_pipeline"):
            pipeline_config = config.full_rag_pipeline
            logger.info(f"成功加载配置: {list(pipeline_config.keys())}")
        else:
            logger.warning("未找到full_rag_pipeline配置")

    except Exception as e:
        logger.error(f"组件状态测试失败: {e}")


def main():
    """主函数"""
    logger = get_logger(__name__)
    logger.info("开始完整RAG流水线示例")

    # 测试组件状态
    test_pipeline_components()

    # 测试BM25流水线（不依赖外部服务）
    test_bm25_pipeline()

    # 测试ES流水线（需要ES服务，注释掉避免错误）
    # test_es_pipeline()

    # 测试混合检索流水线（需要ES服务，注释掉避免错误）
    # test_hybrid_pipeline()

    logger.info("完整RAG流水线示例完成")


if __name__ == "__main__":
    main()
