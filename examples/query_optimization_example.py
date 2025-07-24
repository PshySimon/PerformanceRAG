"""
查询优化组件使用示例
"""

from rag.components.query import QueryFactory
from rag.pipeline.registry import ComponentRegistry
from utils.logger import get_logger, setup_logging

setup_logging(level="INFO")
logger = get_logger(__name__)


def test_individual_components():
    """测试单个查询优化组件"""

    # 测试数据
    test_queries = [
        "如何学习Python编程？",
        "苹果公司2023年的财务报告和市场表现如何？",
        "机器学习和深度学习的区别是什么，以及它们的应用场景？",
        "这个问题很复杂",  # 模糊查询
        "AI",  # 简短查询
    ]

    # 测试查询扩展
    logger.info("=== 查询扩展测试 ===")
    expansion_config = {
        "expansion_type": "multi_query",
        "num_queries": 3,
        "debug": True,
    }
    expansion_component = QueryFactory.create_component(
        "expansion", "test_expansion", expansion_config
    )
    expansion_component.initialize()

    for query in test_queries[:2]:
        logger.info(f"\n原始查询: {query}")
        result = expansion_component.process({"query": query})
        logger.info(f"扩展结果: {result.get('expanded_queries', [])}")

    # 测试查询分解
    logger.info("\n=== 查询分解测试 ===")
    decomposition_config = {"max_subqueries": 4, "min_subqueries": 2, "debug": True}
    decomposition_component = QueryFactory.create_component(
        "decomposition", "test_decomposition", decomposition_config
    )
    decomposition_component.initialize()

    for query in test_queries[1:3]:
        logger.info(f"\n原始查询: {query}")
        result = decomposition_component.process({"query": query})
        logger.info(f"分解结果: {result.get('subqueries', [])}")
        logger.info(f"是否需要分解: {result.get('decomposition_needed', False)}")

    # 测试查询消歧
    logger.info("\n=== 查询消歧测试 ===")
    disambiguation_config = {"ambiguity_threshold": 0.5, "debug": True}
    disambiguation_component = QueryFactory.create_component(
        "disambiguation", "test_disambiguation", disambiguation_config
    )
    disambiguation_component.initialize()

    for query in test_queries[3:5]:
        logger.info(f"\n原始查询: {query}")
        result = disambiguation_component.process({"query": query})
        logger.info(f"消歧结果: {result.get('disambiguated_query', '')}")
        logger.info(f"歧义分数: {result.get('ambiguity_score', 0)}")

    # 测试查询抽象
    logger.info("\n=== 查询抽象测试 ===")
    abstraction_config = {
        "abstraction_level": "medium",
        "preserve_domain": True,
        "debug": True,
    }
    abstraction_component = QueryFactory.create_component(
        "abstraction", "test_abstraction", abstraction_config
    )
    abstraction_component.initialize()

    for query in test_queries[1:3]:
        logger.info(f"\n原始查询: {query}")
        result = abstraction_component.process({"query": query})
        logger.info(f"抽象结果: {result.get('abstracted_query', '')}")
        logger.info(f"是否需要抽象: {result.get('abstraction_needed', False)}")


def test_query_pipeline():
    """测试查询优化流水线"""
    logger.info("\n=== 查询优化流水线测试 ===")

    # 创建流水线配置
    pipeline_config = {
        "pipeline": [
            {
                "type": "expansion",
                "name": "multi_query_expansion",
                "config": {"expansion_type": "multi_query", "num_queries": 3},
            },
            {
                "type": "disambiguation",
                "name": "query_disambiguation",
                "config": {"ambiguity_threshold": 0.6},
            },
        ]
    }

    # 创建流水线
    pipeline = QueryFactory.create_pipeline(pipeline_config)

    if pipeline:
        test_query = "这个AI技术怎么样？"
        logger.info(f"\n测试查询: {test_query}")

        result = pipeline.execute({"query": test_query})
        logger.info("\n流水线处理结果:")
        for key, value in result.items():
            if key != "query":
                logger.info(f"  {key}: {value}")
    else:
        logger.info("流水线创建失败")


def test_component_registry():
    """测试组件注册"""
    logger.info("\n=== 组件注册测试 ===")

    # 列出所有注册的查询组件
    query_components = ComponentRegistry.list_components("query")
    logger.info(f"已注册的查询组件: {list(query_components.keys())}")

    # 通过注册表创建组件
    try:
        ExpansionComponent = ComponentRegistry.get("query", "expansion")
        component = ExpansionComponent("test", {"expansion_type": "rewrite"})
        logger.info(f"通过注册表创建组件成功: {component.name}")
    except Exception as e:
        logger.info(f"通过注册表创建组件失败: {e}")


if __name__ == "__main__":
    # 测试单个组件
    test_individual_components()

    # 测试流水线
    test_query_pipeline()

    # 测试组件注册
    test_component_registry()

    logger.info("\n=== 所有测试完成 ===")
