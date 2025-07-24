#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ES索引Pipeline使用示例

本示例展示如何使用ES索引Pipeline：
1. 加载文档
2. 分割文档
3. 索引到Elasticsearch
"""

import os
import sys

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.pipeline.factory import clear_cache, create_pipeline
from utils.config import config
from utils.logger import get_logger, setup_logging

setup_logging(level="DEBUG")


def test_es_index_pipeline():
    """测试ES索引Pipeline"""
    logger = get_logger(__name__)
    logger.info("=== 测试ES索引Pipeline ===")

    try:
        # 创建ES索引Pipeline
        logger.info("创建ES索引Pipeline...")
        pipeline = create_pipeline("es_index_pipeline")
        logger.info(f"Pipeline创建完成，组件列表: {pipeline.list_components()}")

        # 执行索引流程
        logger.info("开始执行索引流程...")
        result = pipeline.run({}, entry_point="document_loader")

        # 检查结果
        if result.get("indexed", False):  # 改为检查 'indexed' 字段
            logger.info(f"索引完成！索引了 {result.get('document_count', 0)} 个文档")  # 改为 'document_count'
            logger.info(f"索引详情: {result.get('metadata', {})}")
        else:
            logger.error(f"索引失败: {result.get('error', '未知错误')}")

        logger.info("ES索引Pipeline测试完成")

    except Exception as e:
        logger.error(f"ES索引Pipeline测试失败: {e}")
        import traceback

        traceback.print_exc()


def check_index_status():
    """检查索引状态"""
    logger = get_logger(__name__)
    logger.info("=== 检查索引状态 ===")

    try:
        from rag.components.index.es_indexer import ESIndexerComponent

        # 获取ES配置
        es_config = config.es_index_pipeline.components.es_indexer.config

        # 创建ES索引器检查状态
        indexer = ESIndexerComponent("status_checker", es_config)
        indexer.initialize()

        # 检查索引是否存在
        index_name = es_config["index_name"]
        if indexer.client.indices.exists(index=index_name):
            # 获取索引统计信息
            stats = indexer.client.indices.stats(index=index_name)
            doc_count = stats["indices"][index_name]["total"]["docs"]["count"]
            size = stats["indices"][index_name]["total"]["store"]["size_in_bytes"]

            logger.info(f"索引 {index_name} 存在")
            logger.info(f"文档数量: {doc_count}")
            logger.info(f"索引大小: {size} bytes")
        else:
            logger.warning(f"索引 {index_name} 不存在")

    except Exception as e:
        logger.error(f"检查索引状态失败: {e}")


def main():
    """主函数"""
    logger = get_logger(__name__)
    logger.info("开始ES索引Pipeline示例")

    # 检查索引状态
    check_index_status()

    print("\n" + "=" * 50 + "\n")

    # 执行索引
    test_es_index_pipeline()

    print("\n" + "=" * 50 + "\n")

    # 再次检查索引状态
    check_index_status()

    # 清理缓存
    clear_cache()
    logger.info("缓存已清理")

    logger.info("ES索引Pipeline示例完成")


if __name__ == "__main__":
    main()
