from .file_loader import FileLoader
from .web_loader import WebLoader
from .base_loader import BaseLoader


def create_loader(config: dict) -> BaseLoader:
    source_type = config.get("type")

    if source_type == "file":
        return FileLoader(
            path=config.get("path"), file_types=config.get("file_types", None)
        )
    elif source_type == "web":
        return WebLoader(urls=config.get("urls", []))
    else:
        raise ValueError(f"不支持的Loader类型: {source_type}")
