from ...pipeline.registry import ComponentRegistry
from .base_generator import BaseGeneratorComponent
from .llm_generator import LLMGeneratorComponent
from .template_generator import TemplateGeneratorComponent

# 注册组件
ComponentRegistry.register("generator", "llm")(LLMGeneratorComponent)
ComponentRegistry.register("generator", "template")(TemplateGeneratorComponent)

__all__ = [
    "BaseGeneratorComponent",
    "LLMGeneratorComponent",
    "TemplateGeneratorComponent"
]