from abc import abstractmethod
from typing import Any, Dict

from utils.llm.factory import LLMFactory

from ..base import Component


class BaseQueryComponent(Component):
    """查询优化组件基类"""

    def __init__(self, name: str, config: Dict[str, Any]):
        super().__init__(name, config)
        self.llm = None
        self.max_retries = config.get("max_retries", 3)
        self.temperature = config.get("temperature", 0.7)

    def _do_initialize(self):
        """初始化LLM"""
        try:
            self.llm = LLMFactory.from_config()
            self.logger.info(f"查询组件 {self.name} 初始化成功")
        except Exception as e:
            self.logger.error(f"查询组件 {self.name} 初始化失败: {e}")
            raise

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """处理查询数据"""
        query = data.get("query", "")
        if not query:
            self.logger.warning("输入数据中没有找到query字段")
            return data

        try:
            # 执行查询优化
            optimized_result = self._optimize_query(query)

            # 更新数据
            result = data.copy()
            result.update(optimized_result)

            if self.debug:
                self.logger.debug(f"查询优化结果: {optimized_result}")

            return result

        except Exception as e:
            self.logger.error(f"查询优化失败: {e}")
            # 返回原始数据，不中断流程
            return data

    @abstractmethod
    def _optimize_query(self, query: str) -> Dict[str, Any]:
        """执行具体的查询优化逻辑"""
        pass

    def _extract_output(self, response: str) -> str:
        """从LLM响应中提取输出内容"""
        import re

        match = re.search(r"<output>(.*?)</output>", response, re.DOTALL)
        return match.group(1).strip() if match else response.strip()

    def _call_llm_with_retry(self, prompt: str) -> str:
        """带重试的LLM调用"""
        for attempt in range(self.max_retries):
            try:
                response = self.llm.completion(prompt, temperature=self.temperature)
                return response
            except Exception as e:
                self.logger.warning(
                    f"LLM调用失败 (尝试 {attempt + 1}/{self.max_retries}): {e}"
                )
                if attempt == self.max_retries - 1:
                    raise
        return ""
