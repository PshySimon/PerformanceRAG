from typing import Dict, Any, Optional

from ...components.base import Component
from .loader_utils import WebLoader
from utils.logger import get_logger

class WebLoaderComponent(Component):
    """网页加载组件，基于WebLoader实现"""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        super().__init__(name, config)
        self.logger = get_logger(__name__)
        self.urls = config.get("urls", [])
    
    def _do_initialize(self):
        """初始化加载器"""
        if not self.urls:
            self.logger.warning("未配置任何URL")
    
    def get_data_length(self, data: Dict[str, Any]) -> Optional[int]:
        """获取数据长度 - 返回要处理的URL总数"""
        # 如果输入数据中已有文档，返回文档数量
        if "documents" in data and data["documents"]:
            return len(data["documents"])
        
        # 否则返回要加载的URL总数
        urls = data.get("urls", self.urls)
        return len(urls) if urls else 0
    
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """处理输入数据，加载网页文档"""
        # 如果已经有输入文档，则直接使用
        if "documents" in data and data["documents"]:
            self.logger.info(f"使用输入的{len(data['documents'])}个文档")
            return data
        
        # 从URLs中获取文档
        urls = data.get("urls", self.urls)
        if not urls:
            self.logger.warning("未提供任何URL，无法加载文档")
            result = data.copy()
            result["documents"] = []
            result["total_files"] = 0  # 添加总数信息
            return result
            
        self.logger.info(f"从{len(urls)}个URL加载文档中...")
        loader = WebLoader(urls)
        documents = loader.load()
        
        # 更新结果
        result = data.copy()
        result["documents"] = documents
        result["urls"] = urls
        result["total_files"] = len(urls)  # 添加总数信息
        
        self.logger.info(f"成功加载{len(documents)}个文档")
        return result