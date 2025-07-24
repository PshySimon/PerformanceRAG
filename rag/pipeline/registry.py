from typing import Dict, Type
from ..components.base import Component

class ComponentRegistry:
    """组件注册表，用于注册和获取组件类"""
    
    _registry: Dict[str, Dict[str, Type[Component]]] = {}
    
    @classmethod
    def register(cls, component_type: str, name: str):
        """注册组件的装饰器"""
        def decorator(component_class: Type[Component]):
            if component_type not in cls._registry:
                cls._registry[component_type] = {}
            cls._registry[component_type][name] = component_class
            return component_class
        return decorator
    
    @classmethod
    def get(cls, component_type: str, name: str) -> Type[Component]:
        """获取组件类"""
        if component_type not in cls._registry or name not in cls._registry[component_type]:
            raise ValueError(f"未找到组件: {component_type}.{name}")
        return cls._registry[component_type][name]
    
    @classmethod
    def list_components(cls, component_type: str = None) -> Dict:
        """列出所有注册的组件"""
        if component_type:
            return cls._registry.get(component_type, {})
        return cls._registry