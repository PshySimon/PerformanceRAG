#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pipeline Builder使用示例

本示例展示如何使用PipelineBuilder：
1. 使用Builder构建Pipeline
2. 执行Pipeline
3. 管理Pipeline缓存
"""

import os
import sys

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.pipeline.builder import PipelineBuilder
from rag.pipeline.factory import create_pipeline, build_pipeline, clear_cache
from utils.logger import get_logger, setup_logging

setup_logging(level="DEBUG")


def test_pipeline_builder():
    """测试Pipeline Builder"""
    logger = get_logger(__name__)
    logger.info("=== 测试Pipeline Builder ===")
    
    try:
        # 方式1：使用Builder手动构建
        logger.info("\n--- 使用Builder手动构建 ---")
        builder = PipelineBuilder.from_config("bm25_pipeline")
        
        # 分步构建
        builder.load_config()
        logger.info(f"组件列表: {builder.list_components()}")
        
        builder.validate_config()
        logger.info("配置验证通过")
        
        pipeline = builder.build()
        logger.info(f"Pipeline构建完成，入口点: {pipeline.get_entry_points()}")
        
        # 方式2：使用Factory快速创建
        logger.info("\n--- 使用Factory快速创建 ---")
        pipeline2 = create_pipeline("bm25_pipeline")
        logger.info(f"Pipeline创建完成: {pipeline2.list_components()}")
        
        # 方式3：获取Builder实例（不执行build）
        logger.info("\n--- 获取Builder实例 ---")
        builder2 = build_pipeline("generation_pipeline")
        logger.info(f"Builder创建完成: {builder2.pipeline_name}")
        
        logger.info("Pipeline Builder测试完成")
        
    except Exception as e:
        logger.error(f"Pipeline Builder测试失败: {e}")
        import traceback
        traceback.print_exc()


def test_pipeline_execution():
    """测试Pipeline执行"""
    logger = get_logger(__name__)
    logger.info("=== 测试Pipeline执行 ===")
    
    try:
        # 创建Pipeline
        pipeline = create_pipeline("bm25_pipeline")
        
        # 执行索引流程
        logger.info("执行索引流程...")
        index_result = pipeline.run({}, entry_point="document_loader")
        logger.info(f"索引结果: {index_result.get('metadata', {})}")
        
        # 执行检索流程
        logger.info("执行检索流程...")
        query_result = pipeline.run({
            "query": "什么是perf",
            "top_k": 3
        }, entry_point="bm25_retriever")
        logger.info(f"检索到 {query_result.get('result_count', 0)} 个结果")
        
        logger.info("Pipeline执行测试完成")
        
    except Exception as e:
        logger.error(f"Pipeline执行测试失败: {e}")
        import traceback
        traceback.print_exc()


def main():
    """主函数"""
    logger = get_logger(__name__)
    logger.info("开始Pipeline Builder测试")
    
    # 测试Builder
    test_pipeline_builder()
    
    print("\n" + "="*50 + "\n")
    
    # 测试执行
    test_pipeline_execution()
    
    # 清理缓存
    clear_cache()
    logger.info("缓存已清理")
    
    logger.info("Pipeline Builder测试完成")


if __name__ == "__main__":
    main()