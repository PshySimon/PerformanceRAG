#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generation组件使用示例

本示例展示如何使用Generation组件：
1. 创建并测试LLM生成器
2. 创建并测试模板生成器
3. 集成到完整的RAG流水线中
"""

import os
import sys

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.components.generation.llm_generator import LLMGeneratorComponent
from rag.components.generation.template_generator import TemplateGeneratorComponent
from utils.logger import get_logger, setup_logging

setup_logging(level="DEBUG")


def test_template_generator():
    """测试模板生成器"""
    logger = get_logger(__name__)
    logger.info("=== 测试模板生成器 ===")

    try:
        # 创建模板生成器
        generator = TemplateGeneratorComponent(
            name="test_template_generator",
            config={
                "template_name": "retrieval_prompt",
                "max_context_length": 1000,
                "debug": True,
            },
        )

        # 初始化
        generator.initialize()

        # 模拟检索结果
        mock_context = [
            {
                "content": "Linux性能分析工具Perf是一个强大的性能分析工具，可以用来分析CPU使用情况、内存访问模式等。",
                "metadata": {"title": "Perf工具介绍", "category": "Linux Tools"},
                "score": 0.95,
            },
            {
                "content": "使用perf命令可以进行系统级性能分析，包括CPU缓存命中率、分支预测等指标的监控。",
                "metadata": {
                    "title": "Perf命令使用",
                    "category": "Performance Analysis",
                },
                "score": 0.87,
            },
        ]

        # 测试生成
        test_query = "如何使用perf工具进行性能分析？"
        result = generator.process({"query": test_query, "results": mock_context})

        logger.info(f"查询: {test_query}")
        logger.info(f"生成的回答:\n{result['answer']}")
        logger.info(f"元数据: {result['metadata']}")

        logger.info("模板生成器测试完成")

    except Exception as e:
        logger.error(f"模板生成器测试失败: {e}")
        import traceback

        traceback.print_exc()


def test_llm_generator():
    """测试LLM生成器（需要配置API密钥）"""
    logger = get_logger(__name__)
    logger.info("=== 测试LLM生成器 ===")

    try:
        # 创建LLM生成器（使用模拟配置）
        LLMGeneratorComponent(
            name="test_llm_generator",
            config={
                "provider": "openai",
                "model_name": "gpt-3.5-turbo",
                "api_key": "your-api-key",  # 需要替换为真实的API密钥
                "temperature": 0.7,
                "max_tokens": 512,
                "prompt_template": "retrieval_prompt",
                "system_prompt": "你是一个专业的技术文档助手。",
                "debug": True,
            },
        )

        # 注意：这里不初始化，因为需要真实的API密钥
        logger.info("LLM生成器创建成功（未初始化，需要真实API密钥）")

        # 如果有真实的API密钥，可以取消注释以下代码进行测试
        # generator.initialize()
        #
        # mock_context = [...] # 同上
        # test_query = "如何使用perf工具进行性能分析？"
        # result = generator.process({
        #     "query": test_query,
        #     "results": mock_context
        # })
        #
        # logger.info(f"查询: {test_query}")
        # logger.info(f"生成的回答: {result['answer']}")

        logger.info("LLM生成器测试完成（跳过实际调用）")

    except Exception as e:
        logger.error(f"LLM生成器测试失败: {e}")
        import traceback

        traceback.print_exc()


def test_full_rag_pipeline():
    """测试完整的RAG流水线（包含生成）"""
    logger = get_logger(__name__)
    logger.info("=== 测试完整RAG流水线 ===")

    try:
        # 创建流水线（需要先创建配置文件）
        # pipeline = create_pipeline("generation_pipeline")
        # pipeline.build()

        logger.info("完整RAG流水线测试需要配置文件支持")
        logger.info("请参考 config/generation_pipeline.yaml 配置示例")

    except Exception as e:
        logger.error(f"完整RAG流水线测试失败: {e}")
        import traceback

        traceback.print_exc()


def main():
    """主函数"""
    logger = get_logger(__name__)
    logger.info("开始Generation组件测试")

    # 测试模板生成器
    test_template_generator()

    print("\n" + "=" * 50 + "\n")

    # 测试LLM生成器
    test_llm_generator()

    print("\n" + "=" * 50 + "\n")

    # 测试完整流水线
    test_full_rag_pipeline()

    logger.info("Generation组件测试完成")


if __name__ == "__main__":
    main()
