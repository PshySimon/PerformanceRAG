from abc import abstractmethod
from typing import Any, Dict, List, Iterator
from rag.components.base import Component
from utils.logger import get_logger

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
    def generate(self, query: str, context: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        """根据查询和上下文生成回答"""
        pass
    
    @abstractmethod
    def generate_stream(self, query: str, context: List[Dict[str, Any]], **kwargs) -> Iterator[str]:
        """流式生成回答"""
        pass

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """处理输入数据"""
        if "query" not in data:
            raise ValueError("输入数据必须包含 'query' 字段")

        query = data["query"]
        context = data.get("documents", [])  # 改为 documents

        # 执行生成
        result = self.generate(query, context)

        return {
            "answer": result.get("answer", ""),
            "query": query,
            "context_used": len(context),
            "documents": context,  # 保留文档信息
            "metadata": {
                "component": self.name,
                "generator_type": self.__class__.__name__,
                "model_name": self.model_name,
                "temperature": self.temperature,
                **result.get("metadata", {}),
            },
        }
    
    def process_stream(self, data: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
        """流式处理输入数据"""
        if "query" not in data:
            raise ValueError("输入数据必须包含 'query' 字段")

        query = data["query"]
        context = data.get("documents", [])
        
        # 先返回基础信息
        base_result = {
            "query": query,
            "context_used": len(context),
            "documents": context,
            "metadata": {
                "component": self.name,
                "generator_type": self.__class__.__name__,
                "model_name": self.model_name,
                "temperature": self.temperature,
            },
        }
        
        # 流式生成答案
        answer_chunks = []
        for chunk in self.generate_stream(query, context):
            answer_chunks.append(chunk)
            result = base_result.copy()
            result["answer_chunk"] = chunk
            result["answer_partial"] = "".join(answer_chunks)
            yield result
        
        # 最终结果
        final_result = base_result.copy()
        final_result["answer"] = "".join(answer_chunks)
        final_result["is_final"] = True
        yield final_result
