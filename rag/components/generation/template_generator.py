from typing import Any, Dict, List
from .base_generator import BaseGeneratorComponent
from utils.prompt import prompt_manager


class TemplateGeneratorComponent(BaseGeneratorComponent):
    """基于模板的简单生成器组件"""

    def __init__(self, name: str, config: Dict[str, Any]):
        super().__init__(name, config)
        self.template_name = config.get("template_name", "retrieval_prompt")
        self.max_context_length = config.get("max_context_length", 2000)

    def _do_initialize(self):
        """初始化模板生成器"""
        try:
            # 验证模板是否存在
            prompt_manager.get_template(self.template_name)
            
            if self.debug:
                self.logger.debug(f"模板生成器初始化完成，模板: {self.template_name}")
                
        except Exception as e:
            self.logger.error(f"初始化模板生成器失败: {e}")
            raise

    def _format_context(self, context: List[Dict[str, Any]]) -> str:
        """格式化上下文"""
        if not context:
            return "暂无相关文档。"
            
        formatted_docs = []
        total_length = 0
        
        for i, doc in enumerate(context, 1):
            content = doc.get("content", "")
            metadata = doc.get("metadata", {})
            title = metadata.get("title", f"文档{i}")
            
            doc_text = f"文档{i} - {title}:\n{content}"
            
            # 检查长度限制
            if total_length + len(doc_text) > self.max_context_length:
                # 截断内容
                remaining_length = self.max_context_length - total_length
                if remaining_length > 100:  # 至少保留100字符
                    doc_text = doc_text[:remaining_length] + "..."
                    formatted_docs.append(doc_text)
                break
            
            formatted_docs.append(doc_text)
            total_length += len(doc_text)
            
        return "\n\n".join(formatted_docs)

    def generate(
        self, 
        query: str, 
        context: List[Dict[str, Any]], 
        **kwargs
    ) -> Dict[str, Any]:
        """生成回答"""
        try:
            # 获取模板
            template = prompt_manager.get_template(self.template_name)
            
            # 格式化上下文
            formatted_context = self._format_context(context)
            
            # 填充模板
            answer = template.fill(
                documents=formatted_context,
                question=query
            )
            
            return {
                "answer": answer,
                "metadata": {
                    "template_used": self.template_name,
                    "context_length": len(formatted_context),
                    "context_count": len(context)
                }
            }
            
        except Exception as e:
            self.logger.error(f"模板生成失败: {e}")
            
            # 返回简单的拼接结果
            simple_answer = f"""根据检索到的{len(context)}个文档，针对问题"{query}"的回答：

{self._format_context(context)}

请根据以上信息回答问题。"""
            
            return {
                "answer": simple_answer,
                "metadata": {
                    "fallback": True,
                    "error": str(e)
                }
            }