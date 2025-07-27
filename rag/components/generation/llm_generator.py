from typing import Any, Dict, List, Iterator

from utils.llm.factory import LLMFactory
from utils.prompt import prompt_manager

from .base_generator import BaseGeneratorComponent

class LLMGeneratorComponent(BaseGeneratorComponent):
    """基于LLM的生成器组件"""

    def __init__(self, name: str, config: Dict[str, Any]):
        super().__init__(name, config)

        # 提示词模板配置
        self.prompt_template = config.get("prompt_template", "retrieval_prompt")
        self.system_prompt = config.get("system_prompt", "你是一个有用的AI助手。")

        # LLM客户端
        self.llm_client = None

    def _do_initialize(self):
        """初始化生成器"""
        try:
            # 从utils.llm模块获取LLM客户端
            self.llm_client = LLMFactory.from_config()

            if self.debug:
                self.logger.debug(
                    f"LLM生成器初始化完成，使用模型: {getattr(self.llm_client, 'model', 'unknown')}"
                )

        except Exception as e:
            self.logger.error(f"初始化LLM生成器失败: {e}")
            raise

    def _format_context(self, context: List[Dict[str, Any]]) -> str:
        """格式化上下文文档"""
        if not context:
            return "暂无相关文档。"

        formatted_docs = []
        for i, doc in enumerate(context, 1):
            content = doc.get("content", "")
            metadata = doc.get("metadata", {})
            title = metadata.get("title", f"文档{i}")

            formatted_docs.append(f"文档{i} - {title}:\n{content}")

        return "\n\n".join(formatted_docs)

    def _build_prompt(self, query: str, context: List[Dict[str, Any]]) -> str:
        """构建提示词"""
        try:
            # 使用prompt_manager获取模板
            template = prompt_manager.get_template(self.prompt_template)

            # 格式化上下文
            formatted_context = self._format_context(context)

            # 填充模板
            prompt = template.fill(documents=formatted_context, question=query)

            return prompt

        except Exception as e:
            self.logger.warning(f"使用模板失败，使用默认格式: {e}")

            # 默认格式
            formatted_context = self._format_context(context)
            return f"""基于以下检索到的文档片段，回答用户问题：

文档片段：
{formatted_context}

用户问题：{query}

请提供准确、详细的答案："""

    def generate(self, query: str, context: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        """生成回答"""
        try:
            # 构建提示词
            prompt = self._build_prompt(query, context)
            if self.debug:
                self.logger.debug(f"生成提示词: {prompt[:200]}...")

            # 调用LLM客户端生成回答
            answer = self.llm_client.completion(prompt, **kwargs)

            return {
                "answer": answer,
                "metadata": {
                    "prompt_length": len(prompt),
                    "context_count": len(context),
                    "model": getattr(self.llm_client, "model", "unknown"),
                },
            }

        except Exception as e:
            self.logger.error(f"生成回答失败: {e}")
            return {
                "answer": f"抱歉，生成回答时出现错误: {str(e)}",
                "metadata": {"error": str(e)},
            }
    
    def generate_stream(self, query: str, context: List[Dict[str, Any]], **kwargs) -> Iterator[str]:
        """流式生成回答"""
        try:
            # 构建提示词
            prompt = self._build_prompt(query, context)
            if self.debug:
                self.logger.debug(f"流式生成提示词: {prompt[:200]}...")

            # 调用LLM客户端流式生成回答
            for chunk in self.llm_client.completion_stream(prompt, **kwargs):
                yield chunk

        except Exception as e:
            self.logger.error(f"流式生成回答失败: {e}")
            yield f"抱歉，生成回答时出现错误: {str(e)}"
