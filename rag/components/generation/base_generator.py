from abc import abstractmethod
from typing import Any, Dict, List

from utils.logger import get_logger

from ..base import Component


class BaseGeneratorComponent(Component):
    """生成器组件基类"""

    def __init__(self, name: str, config: Dict[str, Any]):
        super().__init__(name, config)
        self.model_name = config.get("model_name", "default")
        self.temperature = config.get("temperature", 0.7)
        self.max_tokens = config.get("max_tokens", 1024)
        self.system_prompt = config.get("system_prompt", "")
        self.logger = get_logger(__name__)

    @abstractmethod
    def generate(
        self, query: str, context: List[Dict[str, Any]], **kwargs
    ) -> Dict[str, Any]:
        """根据查询和上下文生成回答"""
        pass

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """处理输入数据"""
        if "query" not in data:
            raise ValueError("输入数据必须包含 'query' 字段")

        query = data["query"]
        context = data.get("results", [])

        # 执行生成
        result = self.generate(query, context)

        return {
            "answer": result.get("answer", ""),
            "query": query,
            "context_used": len(context),
            "metadata": {
                "component": self.name,
                "generator_type": self.__class__.__name__,
                "model_name": self.model_name,
                "temperature": self.temperature,
                **result.get("metadata", {}),
            },
        }
