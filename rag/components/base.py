from abc import ABC, abstractmethod
from typing import Any, Dict, Iterator

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

    def add_next(self, component: "Component"):
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

    def process_stream(self, data: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
        """流式处理数据，默认实现为非流式"""
        yield self.process(data)

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

        # 传递给下一个组件
        return self._next_components[0].execute(result)

    def execute_stream(self, data: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
        """流式执行当前组件并传递给下一个组件"""
        if not self._initialized:
            self.initialize()

        if self.debug:
            self.logger.debug(f"组件 {self.name} 开始流式处理数据")

        # 流式处理当前组件
        for result in self.process_stream(data):
            if self.debug:
                self.logger.debug(f"组件 {self.name} 产生一个结果")
            
            # 如果没有下一个组件，直接返回结果
            if not self._next_components:
                if self.debug:
                    self.logger.debug(f"组件 {self.name} 是终点组件，返回结果")
                yield result
            else:
                # 传递给下一个组件进行流式处理
                for next_result in self._next_components[0].execute_stream(result):
                    yield next_result

        if self.debug:
            self.logger.debug(f"组件 {self.name} 流式处理完成")