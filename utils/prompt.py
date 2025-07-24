"""
Prompt 模板处理工具
提供简单方便的模板填充功能
"""

import re
from typing import Dict, List, Optional, Union

from .common import ALL_TEMPLATES


class PromptTemplate:
    """Prompt 模板类"""

    def __init__(self, template: str, name: Optional[str] = None):
        """
        初始化模板

        Args:
            template: 模板字符串
            name: 模板名称
        """
        self.template = template
        self.name = name
        self._variables = self._extract_variables()

    def _extract_variables(self) -> List[str]:
        """提取模板中的变量"""
        pattern = r"\{([^}]+)\}"
        return list(set(re.findall(pattern, self.template)))

    @property
    def variables(self) -> List[str]:
        """获取模板变量列表"""
        return self._variables.copy()

    def fill(self, **kwargs) -> str:
        """
        填充模板

        Args:
            **kwargs: 模板变量的值

        Returns:
            填充后的字符串

        Raises:
            ValueError: 当缺少必需变量时
        """
        missing_vars = set(self._variables) - set(kwargs.keys())
        if missing_vars:
            raise ValueError(f"缺少必需的模板变量: {missing_vars}")

        try:
            return self.template.format(**kwargs)
        except KeyError as e:
            raise ValueError(f"模板变量错误: {e}")

    def partial_fill(self, **kwargs) -> "PromptTemplate":
        """
        部分填充模板，返回新的模板对象

        Args:
            **kwargs: 要填充的变量

        Returns:
            新的 PromptTemplate 对象
        """
        # 只填充提供的变量
        available_vars = {k: v for k, v in kwargs.items() if k in self._variables}

        if not available_vars:
            return PromptTemplate(self.template, self.name)

        try:
            # 使用部分格式化
            partial_template = self.template
            for key, value in available_vars.items():
                partial_template = partial_template.replace(f"{{{key}}}", str(value))

            return PromptTemplate(partial_template, self.name)
        except Exception as e:
            raise ValueError(f"部分填充失败: {e}")

    def validate(self, **kwargs) -> bool:
        """
        验证提供的参数是否满足模板要求

        Args:
            **kwargs: 要验证的参数

        Returns:
            是否满足要求
        """
        return set(self._variables).issubset(set(kwargs.keys()))

    def __str__(self) -> str:
        return f"PromptTemplate(name='{self.name}', variables={self.variables})"

    def __repr__(self) -> str:
        return self.__str__()


class PromptManager:
    """Prompt 管理器"""

    def __init__(self):
        """初始化管理器"""
        self._templates: Dict[str, PromptTemplate] = {}
        self._load_default_templates()

    def _load_default_templates(self):
        """加载默认模板"""
        for name, template_str in ALL_TEMPLATES.items():
            self._templates[name] = PromptTemplate(template_str, name)

    def register_template(self, name: str, template: Union[str, PromptTemplate]):
        """
        注册新模板

        Args:
            name: 模板名称
            template: 模板字符串或 PromptTemplate 对象
        """
        if isinstance(template, str):
            template = PromptTemplate(template, name)
        elif isinstance(template, PromptTemplate):
            template.name = name
        else:
            raise ValueError("template 必须是字符串或 PromptTemplate 对象")

        self._templates[name] = template

    def get_template(self, name: str) -> PromptTemplate:
        """
        获取模板

        Args:
            name: 模板名称

        Returns:
            PromptTemplate 对象

        Raises:
            KeyError: 当模板不存在时
        """
        if name not in self._templates:
            raise KeyError(f"模板 '{name}' 不存在")
        return self._templates[name]

    def list_templates(self) -> List[str]:
        """列出所有模板名称"""
        return list(self._templates.keys())

    def fill_template(self, name: str, **kwargs) -> str:
        """
        直接填充指定模板

        Args:
            name: 模板名称
            **kwargs: 模板变量

        Returns:
            填充后的字符串
        """
        template = self.get_template(name)
        return template.fill(**kwargs)

    def remove_template(self, name: str):
        """删除模板"""
        if name in self._templates:
            del self._templates[name]


# 全局管理器实例
prompt_manager = PromptManager()


# ==================== 便捷函数 ====================


def create_template(template: str, name: Optional[str] = None) -> PromptTemplate:
    """
    创建模板

    Args:
        template: 模板字符串
        name: 模板名称

    Returns:
        PromptTemplate 对象
    """
    return PromptTemplate(template, name)


def fill_template(template: Union[str, PromptTemplate], **kwargs) -> str:
    """
    填充模板（便捷函数）

    Args:
        template: 模板字符串或 PromptTemplate 对象
        **kwargs: 模板变量

    Returns:
        填充后的字符串
    """
    if isinstance(template, str):
        template = PromptTemplate(template)
    return template.fill(**kwargs)


def get_template(name: str) -> PromptTemplate:
    """
    获取预定义模板

    Args:
        name: 模板名称

    Returns:
        PromptTemplate 对象
    """
    return prompt_manager.get_template(name)


def quick_fill(template_name: str, **kwargs) -> str:
    """
    快速填充预定义模板

    Args:
        template_name: 模板名称
        **kwargs: 模板变量

    Returns:
        填充后的字符串
    """
    return prompt_manager.fill_template(template_name, **kwargs)


def list_available_templates() -> List[str]:
    """列出所有可用模板"""
    return prompt_manager.list_templates()


# ==================== 特殊用途函数 ====================


def format_documents(documents: List[str], separator: str = "\n\n---\n\n") -> str:
    """
    格式化文档列表为字符串

    Args:
        documents: 文档列表
        separator: 分隔符

    Returns:
        格式化后的字符串
    """
    return separator.join(f"文档 {i+1}:\n{doc}" for i, doc in enumerate(documents))


def create_rag_prompt(
    question: str, documents: List[str], template_name: str = "retrieval_prompt"
) -> str:
    """
    创建 RAG 提示词

    Args:
        question: 用户问题
        documents: 检索到的文档
        template_name: 使用的模板名称

    Returns:
        完整的 RAG 提示词
    """
    formatted_docs = format_documents(documents)
    return quick_fill(template_name, question=question, documents=formatted_docs)


def create_multi_turn_prompt(
    conversation_history: List[Dict[str, str]], current_question: str
) -> str:
    """
    创建多轮对话提示词

    Args:
        conversation_history: 对话历史 [{'role': 'user/assistant', 'content': '...'}]
        current_question: 当前问题

    Returns:
        多轮对话提示词
    """
    history_text = "\n".join(
        [f"{item['role']}: {item['content']}" for item in conversation_history]
    )

    template = (
        "对话历史：\n{history}\n\n"
        "当前问题：{question}\n\n"
        "请基于对话历史回答当前问题："
    )

    return fill_template(template, history=history_text, question=current_question)
