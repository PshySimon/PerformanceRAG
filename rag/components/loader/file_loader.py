from typing import Dict, Any
from pathlib import Path

from ...components.base import Component
from .loader_utils import FileLoader
from utils.logger import get_logger

class FileLoaderComponent(Component):
    """文件加载组件，基于现有的FileLoader实现"""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        super().__init__(name, config)
        self.logger = get_logger(__name__)
        self.path = config.get("path", "./data")
        self.file_types = config.get("file_types", [".txt", ".md", ".pdf", ".html"])
        self._documents = None
    
    def _do_initialize(self):
        """初始化加载器"""
        # 这里可以进行一些初始化工作，如检查路径是否存在等
        path = Path(self.path)
        if not path.exists():
            self.logger.warning(f"路径不存在: {self.path}，将在处理时创建")
    
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """处理输入数据，加载文档"""
        # 如果已经有输入文档，则直接使用
        if "documents" in data and data["documents"]:
            self.logger.info(f"使用输入的{len(data['documents'])}个文档")
            return data
        
        # 否则从文件系统加载
        self.logger.info(f"从{self.path}加载文件中...")
        loader = FileLoader(self.path, self.file_types)
        documents = loader.load()
        
        # 更新结果
        result = data.copy()
        result["documents"] = documents
        result["source_path"] = self.path
        
        self.logger.info(f"成功加载{len(documents)}个文档")
        return result