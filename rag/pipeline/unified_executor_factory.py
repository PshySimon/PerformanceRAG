from typing import Any, Dict, Union
from utils.logger import get_logger
from utils.config import config

from .executor import PipelineExecutor
from .producer_consumer_executor import ProducerConsumerPipelineExecutor
from .async_producer_consumer_executor import AsyncProducerConsumerPipelineExecutor
from .async_datasource_producer_consumer_executor import AsyncDataSourceProducerConsumerPipelineExecutor
from .factory import create_pipeline


class UnifiedExecutorFactory:
    """统一执行器工厂，根据配置自动选择合适的执行器类型"""
    
    def __init__(self):
        self.logger = get_logger(__name__)
    
    def _detect_executor_type(self, config_data: Dict[str, Any]) -> str:
        """根据architecture字段检测执行器类型"""
        
        # 优先检查architecture字段
        if "architecture" in config_data:
            arch_type = config_data["architecture"].get("type")
            if arch_type:
                self.logger.debug(f"从architecture字段检测到类型: {arch_type}")
                return arch_type
        
        # 如果没有architecture字段，使用原有的启发式检测
        # 检查是否有datasource、producer、consumer三个部分
        if all(key in config_data for key in ["datasource", "producer", "consumer"]):
            return "async_datasource_producer_consumer"
        
        # 检查是否有producer、consumer两个部分
        if all(key in config_data for key in ["producer", "consumer"]):
            return "producer_consumer"
        
        # 检查是否是标准的pipeline配置
        if "components" in config_data and "flow" in config_data:
            return "standard_pipeline"
        
        # 默认返回标准pipeline
        return "standard_pipeline"
    
    def create_executor(self, pipeline_name: str, **kwargs) -> Union[
        PipelineExecutor, 
        ProducerConsumerPipelineExecutor, 
        AsyncProducerConsumerPipelineExecutor, 
        AsyncDataSourceProducerConsumerPipelineExecutor
    ]:
        """创建统一执行器
        
        Args:
            pipeline_name: 配置名称
            **kwargs: 额外参数，如use_cache等
            
        Returns:
            对应类型的执行器实例
        """
        try:
            # 从config获取配置数据
            if not hasattr(config, pipeline_name):
                raise ValueError(f"找不到Pipeline配置: {pipeline_name}")
            
            config_data = getattr(config, pipeline_name)
            
            # 检测执行器类型
            executor_type = self._detect_executor_type(config_data)
            
            self.logger.info(f"检测到执行器类型: {executor_type} (配置: {pipeline_name})")
            
            # 根据类型创建对应的执行器
            if executor_type == "async_datasource_producer_consumer":
                return AsyncDataSourceProducerConsumerPipelineExecutor(config_data)
            
            elif executor_type == "async_producer_consumer":
                return AsyncProducerConsumerPipelineExecutor(config_data)
            
            elif executor_type == "producer_consumer":
                return ProducerConsumerPipelineExecutor(config_data)
            
            elif executor_type == "standard_pipeline":
                # 使用现有的factory创建标准pipeline
                use_cache = kwargs.get("use_cache", True)
                return create_pipeline(pipeline_name, use_cache=use_cache)
            
            else:
                raise ValueError(f"不支持的执行器类型: {executor_type}")
                
        except Exception as e:
            self.logger.error(f"创建统一执行器失败: {pipeline_name}, 错误: {e}")
            raise
    
    def get_executor_type(self, pipeline_name: str) -> str:
        """获取指定配置的执行器类型（不创建实例）"""
        if not hasattr(config, pipeline_name):
            raise ValueError(f"找不到Pipeline配置: {pipeline_name}")
        
        config_data = getattr(config, pipeline_name)
        return self._detect_executor_type(config_data)


# 全局实例
_unified_factory = UnifiedExecutorFactory()


def create_unified_executor(pipeline_name: str, **kwargs):
    """创建统一执行器的便捷函数
    
    Args:
        pipeline_name: 配置名称
        **kwargs: 额外参数
        
    Returns:
        自动选择的执行器实例
    """
    return _unified_factory.create_executor(pipeline_name, **kwargs)


def get_executor_type(pipeline_name: str) -> str:
    """获取执行器类型的便捷函数"""
    return _unified_factory.get_executor_type(pipeline_name)