from typing import Dict, Any, Iterator, Optional
from pathlib import Path

from ...components.base import Component
from .loader_utils import FileLoader
from utils.logger import get_logger

class FileLoaderComponent(Component):
    """文件加载组件"""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        super().__init__(name, config)
        self.logger = get_logger(__name__)
        self.path = config.get("path", "./data")
        self.file_types = config.get("file_types", [".txt", ".md", ".pdf", ".html"])
        self._documents = None
        self._total_files = None
    
    def _count_files(self) -> int:
        """统计要处理的文件总数"""
        if self._total_files is not None:
            return self._total_files
            
        path = Path(self.path)
        if not path.exists():
            self._total_files = 0
            return 0
            
        if path.is_file():
            self._total_files = 1
        else:
            # 统计目录下所有符合条件的文件
            all_files = []
            for file_type in self.file_types:
                all_files.extend(list(path.glob(f"**/*{file_type}")))
            self._total_files = len(all_files)
            
        return self._total_files
    
    def _do_initialize(self):
        """初始化加载器"""
        # 预先统计文件数量
        total_files = self._count_files()
        self.logger.info(f"发现 {total_files} 个文件待处理")
        
        path = Path(self.path)
        if not path.exists():
            self.logger.warning(f"路径不存在: {self.path}，将在处理时创建")
    
    def get_data_length(self, input_data: Dict[str, Any]) -> int:
        """获取数据长度 - 返回预期处理的文档总数"""
        if "documents" in input_data:
            return len(input_data["documents"])
        
        # 如果没有预加载文档，需要预估文档数量
        # 这里可以通过快速扫描文件来估算文档数量
        total_files = self._count_files()

        return total_files
    
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """处理模式"""
        # 如果已经有输入文档，则直接使用
        if "documents" in data and data["documents"]:
            self.logger.info(f"使用输入的{len(data['documents'])}个文档")
            return data
        
        # 否则从文件系统加载
        total_files = self._count_files()
        self.logger.info(f"从{self.path}加载 {total_files} 个文件中...")
        loader = FileLoader(self.path, self.file_types)
        documents = loader.load()
        
        # 更新结果
        result = data.copy()
        result["documents"] = documents
        result["source_path"] = self.path
        result["total_files"] = total_files
        
        self.logger.info(f"成功加载{len(documents)}个文档")
        return result
    
    def process_stream(self, data: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
        """流式处理模式 - 逐个文档返回"""
        # 如果已经有输入文档，则直接使用
        if "documents" in data and data["documents"]:
            self.logger.info(f"使用输入的{len(data['documents'])}个文档")
            # 逐个返回文档
            for i, doc in enumerate(data["documents"]):
                result = data.copy()
                result["documents"] = [doc]
                result["batch_index"] = i
                result["total_files"] = len(data["documents"])  # 添加总数信息
                result["is_last_batch"] = (i == len(data["documents"]) - 1)
                yield result
            return
        
        # 否则从文件系统流式加载
        total_files = self._count_files()
        self.logger.info(f"从{self.path}流式加载 {total_files} 个文件中...")
        
        loader = FileLoader(self.path, self.file_types)
        
        # 使用流式加载
        doc_count = 0
        for doc in loader.load_stream():
            result = data.copy()
            result["documents"] = [doc]  # 每次返回一个文档
            result["source_path"] = self.path
            result["total_files"] = total_files
            result["batch_index"] = doc_count
            result["is_last_batch"] = (doc_count == total_files - 1)
            
            doc_count += 1
            yield result
        
        self.logger.info(f"流式加载完成，共处理{doc_count}个文档")
    
