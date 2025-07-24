from .file_loader import FileLoaderComponent
from .web_loader import WebLoaderComponent
from ...pipeline.registry import ComponentRegistry

# 注册组件
ComponentRegistry.register("loader", "file")(FileLoaderComponent)
ComponentRegistry.register("loader", "web")(WebLoaderComponent)

__all__ = ["FileLoaderComponent", "WebLoaderComponent", "BaseLoader", "FileLoader", "WebLoader", "create_loader"]

# 导出加载器工具类
from .base_loader import BaseLoader
from .loader_utils import FileLoader, WebLoader
from .loader_factory import create_loader