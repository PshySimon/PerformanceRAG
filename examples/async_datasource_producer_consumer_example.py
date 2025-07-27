#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
异步数据源-生产者-消费者执行器使用示例

数据源: loader + splitter
生产者: embedding
消费者: indexer
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


async def async_example():
    """异步调用示例"""
    logger = get_logger(__name__)
    logger.info("=== 异步数据源-生产者-消费者Pipeline示例 ===\n")

    # 直接使用 utils.config，无需 dict() 包装
    pipeline_config = config.datasource_producer_consumer_pipeline

    # 创建异步执行器
    executor = AsyncDataSourceProducerConsumerPipelineExecutor(pipeline_config)

    # 异步执行
    start_time = time.time()
    result = await executor.run_async({})
    end_time = time.time()

    logger.info("\n🎯 异步执行完成！")
    logger.info(f"总耗时: {end_time - start_time:.2f}秒")
    logger.info(f"处理结果: {result}")


if __name__ == "__main__":
    # 运行异步示例
    print("运行异步示例...")
    asyncio.run(async_example())

