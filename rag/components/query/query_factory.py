from typing import Any, Dict

from utils.logger import get_logger

from .abstraction_component import AbstractionComponent
from .base_query import BaseQueryComponent
from .decomposition_component import DecompositionComponent
from .disambiguation_component import DisambiguationComponent
from .expansion_component import ExpansionComponent


class QueryFactory:
    """查询组件工厂"""

    _components = {
        "expansion": ExpansionComponent,
        "decomposition": DecompositionComponent,
        "disambiguation": DisambiguationComponent,
        "abstraction": AbstractionComponent,
    }

    @classmethod
    def create_component(
        cls, component_type: str, name: str, config: Dict[str, Any]
    ) -> BaseQueryComponent:
        """创建查询组件"""
        if component_type not in cls._components:
            raise ValueError(f"未知的查询组件类型: {component_type}")

        component_class = cls._components[component_type]
        return component_class(name, config)

    @classmethod
    def create_pipeline(cls, config: Dict[str, Any]) -> BaseQueryComponent:
        """创建查询优化流水线"""
        logger = get_logger(__name__)

        components = []
        pipeline_config = config.get("pipeline", [])

        for step_config in pipeline_config:
            component_type = step_config.get("type")
            component_name = step_config.get("name", component_type)
            component_config = step_config.get("config", {})

            try:
                component = cls.create_component(
                    component_type, component_name, component_config
                )
                components.append(component)
                logger.info(f"创建查询组件: {component_type} - {component_name}")
            except Exception as e:
                logger.error(f"创建查询组件失败: {component_type} - {e}")
                continue

        # 连接组件
        if len(components) > 1:
            for i in range(len(components) - 1):
                components[i].add_next(components[i + 1])

        return components[0] if components else None

    @classmethod
    def list_available_components(cls) -> list:
        """列出所有可用的查询组件类型"""
        return list(cls._components.keys())
