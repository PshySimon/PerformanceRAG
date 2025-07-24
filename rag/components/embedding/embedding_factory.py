from typing import Dict, Any
from .base_embedding import BaseEmbedding
from .openai_embedding import OpenAIEmbedding
from .hf_embedding import HFEmbedding

class EmbeddingFactory:
    """嵌入模型工厂，用于创建不同类型的嵌入模型"""
    
    @staticmethod
    def create(embedding_type: str, **kwargs) -> BaseEmbedding:
        """创建嵌入模型
        
        Args:
            embedding_type: 嵌入模型类型，支持'openai'和'hf'
            **kwargs: 传递给嵌入模型的参数
            
        Returns:
            BaseEmbedding: 嵌入模型实例
            
        Raises:
            ValueError: 不支持的嵌入模型类型
        """
        if embedding_type == "openai":
            return OpenAIEmbedding(**kwargs)
        elif embedding_type == "hf":
            return HFEmbedding(**kwargs)
        else:
            raise ValueError(f"不支持的嵌入模型类型: {embedding_type}")
    
    @staticmethod
    def from_config(config: Dict[str, Any]) -> BaseEmbedding:
        """从配置创建嵌入模型
        
        Args:
            config: 配置字典，必须包含'type'字段
            
        Returns:
            BaseEmbedding: 嵌入模型实例
            
        Raises:
            ValueError: 配置中缺少'type'字段
        """
        if "type" not in config:
            raise ValueError("配置中缺少'type'字段")
        
        embedding_type = config.pop("type")
        return EmbeddingFactory.create(embedding_type, **config)