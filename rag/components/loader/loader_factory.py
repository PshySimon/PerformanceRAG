from .base_loader import BaseLoader
from .loader_utils import FileLoader, WebLoader


def create_loader(config: dict) -> BaseLoader:
    """创建加载器实例

    Args:
        config: 加载器配置

    Returns:
        BaseLoader: 加载器实例
    """
    source_type = config.get("type")

    if source_type == "file":
        return FileLoader(
            path=config.get("path"), file_types=config.get("file_types", None)
        )
    elif source_type == "web":
        return WebLoader(urls=config.get("urls", []))
    else:
        raise ValueError(f"不支持的Loader类型: {source_type}")
