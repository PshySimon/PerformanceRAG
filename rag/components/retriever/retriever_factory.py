from typing import Any, Dict, List

from utils.logger import get_logger

from .bm25_retriever import BM25RetrieverComponent
from .es_retriever import ESRetrieverComponent


class RetrieverFactory:
    """检索器工厂类"""

    @staticmethod
    def create_retriever(retriever_type: str, name: str, config: Dict[str, Any]):
        """创建检索器组件"""
        if retriever_type == "elasticsearch" or retriever_type == "es":
            return ESRetrieverComponent(name, config)
        elif retriever_type == "bm25":
            return BM25RetrieverComponent(name, config)
        else:
            raise ValueError(f"不支持的检索器类型: {retriever_type}")

    @staticmethod
    def create_retriever_pipeline(pipeline_config: Dict[str, Any]) -> List:
        """创建检索器流水线"""
        logger = get_logger(__name__)
        retrievers = []

        for i, retriever_config in enumerate(pipeline_config.get("retrievers", [])):
            retriever_type = retriever_config["type"]
            retriever_name = retriever_config.get("name", f"retriever_{i}")

            try:
                retriever = RetrieverFactory.create_retriever(
                    retriever_type, retriever_name, retriever_config
                )
                retrievers.append(retriever)
                logger.info(f"创建检索器: {retriever_name} ({retriever_type})")
            except Exception as e:
                logger.error(f"创建检索器失败: {e}")
                raise

        return retrievers
