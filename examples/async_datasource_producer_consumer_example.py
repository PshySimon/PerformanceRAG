#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
异步数据源-生产者-消费者执行器使用示例
支持Small2Big检索策略

数据源: loader + splitter (支持层级分割)
生产者: embedding
消费者: indexer (支持层级索引)
"""

import asyncio
import logging
import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.pipeline.async_datasource_producer_consumer_executor import (
    AsyncDataSourceProducerConsumerPipelineExecutor,
)
from rag.components.retriever.es_retriever import ESRetrieverComponent
from utils.config import config
from utils.logger import get_logger, setup_logging

setup_logging(level="INFO")
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("openai._base_client").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("elastic_transport").setLevel(logging.WARNING)
logging.getLogger("elastic_transport.transport").setLevel(logging.WARNING)
logging.getLogger("elasticsearch").setLevel(logging.WARNING)


async def async_indexing_example():
    """异步索引示例 - 支持层级分割和索引"""
    logger = get_logger(__name__)
    logger.info("=== 异步数据源-生产者-消费者Pipeline示例 (支持Small2Big) ===\n")

    # 直接使用 utils.config，无需 dict() 包装
    pipeline_config = config.datasource_producer_consumer_pipeline

    # 创建异步执行器
    executor = AsyncDataSourceProducerConsumerPipelineExecutor(pipeline_config)

    # 异步执行索引
    start_time = time.time()
    result = await executor.run_async({})
    end_time = time.time()

    logger.info("\n🎯 异步索引执行完成！")
    logger.info(f"总耗时: {end_time - start_time:.2f}秒")
    logger.info(f"处理结果: {result}")
    
    return result


async def async_search_example():
    """异步检索示例 - 演示Small2Big检索策略"""
    logger = get_logger(__name__)
    logger.info("\n=== Small2Big检索策略演示 ===\n")
    
    # 加载检索配置
    # 修改第69-70行
    search_config = config.es_search_pipeline
    retriever_config = search_config['components']['es_retriever']['config']
    
    # 创建检索器 - 添加缺少的name参数
    retriever = ESRetrieverComponent("es_retriever", retriever_config)
    retriever.initialize()  # 显式初始化
    
    # 测试查询
    test_queries = [
        "Python异步编程的最佳实践",
        "如何优化Elasticsearch性能",
        "RAG系统的架构设计"
    ]
    
    for i, query in enumerate(test_queries, 1):
        logger.info(f"\n🔍 测试查询 #{i}: {query}")
        
        start_time = time.time()
        results = retriever.retrieve(query, top_k=5)
        end_time = time.time()
        
        logger.info(f"⏱️ 检索耗时: {end_time - start_time:.3f}秒")
        logger.info(f"📊 返回结果数量: {len(results)}")
        
        # 显示检索结果详情
        for j, result in enumerate(results, 1):
            content_preview = result['content'][:100] + "..." if len(result['content']) > 100 else result['content']
            
            logger.info(f"\n📄 结果 #{j}:")
            logger.info(f"   📋 文档ID: {result['id']}")
            logger.info(f"   ⭐ 分数: {result['score']:.4f}")
            logger.info(f"   🏷️ 召回方式: {result['recall_source']}")
            
            if result['recall_source'] == 'small2big':
                logger.info(f"   🔗 来源Small Chunk数: {result.get('source_small_chunks_count', 0)}")
                logger.info(f"   📈 最高Small分数: {result.get('max_small_score', 0):.4f}")
                logger.info(f"   📊 平均Small分数: {result.get('avg_small_score', 0):.4f}")
                logger.info(f"   🔄 融合方法: {result.get('fusion_method', 'unknown')}")
            
            logger.info(f"   📝 内容预览: {content_preview}")
            
            # 显示元数据
            metadata = result.get('metadata', {})
            if metadata:
                logger.info(f"   🏷️ 文件路径: {metadata.get('file_path', 'N/A')}")
                logger.info(f"   📏 Chunk层级: {metadata.get('chunk_level', 'N/A')}")
                logger.info(f"   📦 Chunk大小: {metadata.get('chunk_size', 'N/A')}")


async def main():
    """主函数 - 演示完整的Small2Big流程"""
    logger = get_logger(__name__)
    
    try:
        # 询问用户要执行的操作
        print("\n请选择要执行的操作:")
        print("1. 执行索引 (数据源->生产者->消费者)")
        print("2. 执行检索 (Small2Big策略演示)")
        print("3. 执行完整流程 (索引+检索)")
        
        choice = input("\n请输入选择 (1/2/3): ").strip()
        
        if choice == "1":
            await async_indexing_example()
        elif choice == "2":
            await async_search_example()
        elif choice == "3":
            logger.info("🚀 开始执行完整流程...\n")
            
            # 先执行索引
            await async_indexing_example()
            
            # 等待索引完成
            logger.info("\n⏳ 等待索引刷新...")
            await asyncio.sleep(5)
            
            # 再执行检索
            await async_search_example()
            
            logger.info("\n🎉 完整流程执行完成！")
        else:
            logger.warning("❌ 无效选择，退出程序")
            
    except KeyboardInterrupt:
        logger.info("\n👋 用户中断，程序退出")
    except Exception as e:
        logger.error(f"❌ 程序执行出错: {e}")
        import traceback
        logger.error(traceback.format_exc())


if __name__ == "__main__":
    # 运行主程序
    print("🚀 启动Small2Big检索策略演示程序...")
    asyncio.run(main())

