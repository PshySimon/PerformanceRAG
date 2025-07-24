#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.components.reranker import RerankerFactory
from utils.logger import get_logger


def test_llm_reranker():
    """测试LLM重排组件"""
    logger = get_logger(__name__)
    logger.info("=== LLM重排测试 ===")

    # 创建LLM重排组件
    llm_reranker = RerankerFactory.create_reranker(
        name="test_llm_reranker",
        reranker_type="llm",
        config={"top_k": 5, "method": "listwise", "temperature": 0.3, "debug": True},
    )

    # 初始化组件
    llm_reranker.initialize()

    # 测试数据
    test_data = {
        "query": "什么是机器学习？",
        "documents": [
            {
                "content": "机器学习是人工智能的一个分支，它使计算机能够在没有明确编程的情况下学习。"
            },
            {
                "content": "深度学习是机器学习的一个子集，使用神经网络来模拟人脑的工作方式。"
            },
            {"content": "Python是一种编程语言，广泛用于数据科学和机器学习。"},
            {"content": "数据挖掘是从大量数据中发现模式和知识的过程。"},
            {"content": "自然语言处理是计算机科学和人工智能的一个分支。"},
        ],
    }

    # 执行重排
    result = llm_reranker.process(test_data)

    logger.info(f"重排结果: {len(result['documents'])} 个文档")
    for i, doc in enumerate(result["documents"]):
        logger.info(f"排名 {i+1}: {doc['content'][:50]}...")


def test_embedding_reranker():
    """测试Embedding重排组件"""
    logger = get_logger(__name__)
    logger.info("=== Embedding重排测试 ===")

    # 创建Embedding重排组件
    embedding_reranker = RerankerFactory.create_reranker(
        name="test_embedding_reranker",
        reranker_type="embedding",
        config={
            "top_k": 5,
            "similarity_metric": "cosine",
            "embedding_type": "hf",
            "debug": True,
        },
    )

    # 初始化组件
    embedding_reranker.initialize()

    # 测试数据
    test_data = {
        "query": "机器学习算法",
        "documents": [
            {"content": "支持向量机是一种监督学习算法，用于分类和回归分析。"},
            {"content": "随机森林是一种集成学习方法，结合多个决策树。"},
            {"content": "神经网络是模拟人脑神经元工作的计算模型。"},
            {"content": "线性回归是最简单的机器学习算法之一。"},
            {"content": "聚类算法用于将数据分组到不同的类别中。"},
        ],
    }

    # 执行重排
    result = embedding_reranker.process(test_data)

    logger.info(f"重排结果: {len(result['documents'])} 个文档")
    for i, doc in enumerate(result["documents"]):
        logger.info(f"排名 {i+1}: {doc['content'][:50]}...")


def test_reranker_pipeline():
    """测试重排流水线"""
    logger = get_logger(__name__)
    logger.info("=== 重排流水线测试 ===")

    # 创建流水线配置
    pipeline_config = {
        "components": [
            {
                "name": "embedding_rerank",
                "type": "reranker",
                "subtype": "embedding",
                "config": {
                    "top_k": 8,
                    "similarity_metric": "cosine",
                    "embedding_type": "hf",
                },
            },
            {
                "name": "llm_rerank",
                "type": "reranker",
                "subtype": "llm",
                "config": {"top_k": 5, "method": "listwise", "temperature": 0.3},
            },
        ]
    }

    # 创建重排组件
    rerankers = RerankerFactory.create_pipeline_rerankers(pipeline_config)

    # 初始化所有组件
    for reranker in rerankers.values():
        reranker.initialize()

    # 测试数据
    test_data = {
        "query": "深度学习在计算机视觉中的应用",
        "documents": [
            {"content": "卷积神经网络(CNN)是深度学习在图像识别中的核心技术。"},
            {"content": "目标检测算法如YOLO和R-CNN在计算机视觉中广泛应用。"},
            {"content": "图像分割技术可以精确识别图像中的每个像素。"},
            {"content": "生成对抗网络(GAN)可以生成逼真的图像。"},
            {"content": "迁移学习允许在新任务上重用预训练模型。"},
            {"content": "数据增强技术可以提高模型的泛化能力。"},
            {"content": "注意力机制在视觉任务中越来越重要。"},
            {"content": "自监督学习减少了对标注数据的依赖。"},
        ],
    }

    # 执行流水线重排
    current_data = test_data
    for name, reranker in rerankers.items():
        logger.info(f"执行 {name} 重排...")
        current_data = reranker.process(current_data)
        logger.info(f"{name} 重排后文档数量: {len(current_data['documents'])}")

    logger.info("最终重排结果:")
    for i, doc in enumerate(current_data["documents"]):
        logger.info(f"排名 {i+1}: {doc['content'][:60]}...")


if __name__ == "__main__":
    try:
        test_embedding_reranker()
        print("\n" + "=" * 50 + "\n")
        test_llm_reranker()
        print("\n" + "=" * 50 + "\n")
        test_reranker_pipeline()
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback

        traceback.print_exc()
