from abc import ABC, abstractmethod
from typing import Any, Dict
from utils.logger import get_logger

class Component(ABC):
    """所有Pipeline组件的基类"""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config
        self._next_components = []
        self._initialized = False
        self.debug = config.get("debug", False)
        self.logger = get_logger(__name__)
    
    def add_next(self, component: 'Component'):
        """添加下一个处理组件"""
        self._next_components.append(component)
        return self
    
    def initialize(self):
        """初始化组件资源"""
        if not self._initialized:
            if self.debug:
                self.logger.debug(f"初始化组件: {self.name}")
            self._do_initialize()
            self._initialized = True
    
    @abstractmethod
    def _do_initialize(self):
        """实际的初始化逻辑，由子类实现"""
        pass
    
    @abstractmethod
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """处理输入数据并返回结果，由子类实现"""
        pass
    
    def execute(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """执行当前组件并传递给下一个组件"""
        if not self._initialized:
            self.initialize()
        
        if self.debug:
            self.logger.debug(f"组件 {self.name} 开始处理数据")
            
        result = self.process(data)
        
        if self.debug:
            self.logger.debug(f"组件 {self.name} 处理完成")
        
        # 如果没有下一个组件，直接返回结果
        if not self._next_components:
            if self.debug:
                self.logger.debug(f"组件 {self.name} 是终点组件，返回结果")
            return result
        
        # 如果有多个下一步组件，合并所有结果
        if len(self._next_components) > 1:
            if self.debug:
                self.logger.debug(f"组件 {self.name} 有多个下一步组件，合并结果")
            merged_result = result.copy()
            for component in self._next_components:
                if self.debug:
                    self.logger.debug(f"执行下一步组件: {component.name}")
                next_result = component.execute(result)
                merged_result.update(next_result)
            return merged_result
        
        # 只有一个下一步组件
        if self.debug:
            self.logger.debug(f"执行下一步组件: {self._next_components[0].name}")
        return self._next_components[0].execute(result)