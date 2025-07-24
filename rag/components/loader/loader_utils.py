from typing import List, Dict, Any, Optional
from pathlib import Path

# 修正导入路径
from llama_index.core import SimpleDirectoryReader, Document
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.readers.file import PDFReader
from llama_index.readers.web import SimpleWebPageReader, BeautifulSoupWebReader

from .base_loader import BaseLoader


class FileLoader(BaseLoader):
    """文件加载器，支持加载不同类型的文件"""

    def __init__(self, path: str, file_types: Optional[List[str]] = None, enable_markdown_parsing: bool = True):
        """初始化文件加载器

        Args:
            path: 文件或目录路径
            file_types: 文件类型列表，如['.txt', '.md', '.pdf', '.html']
            enable_markdown_parsing: 是否启用 Markdown 结构化解析
        """
        self.path = path
        self.file_types = file_types or [".txt", ".md", ".pdf", ".html"]
        self.enable_markdown_parsing = enable_markdown_parsing
        
        # 初始化 MarkdownNodeParser
        if self.enable_markdown_parsing:
            self.markdown_parser = MarkdownNodeParser()

    def _extract_heading_context(self, nodes: List[Any]) -> List[Dict[str, Any]]:
        """提取节点的标题上下文，为每个节点添加对应的最小级别标题"""
        documents = []
        current_headings = {}  # 存储当前各级别的标题
        
        for node in nodes:
            metadata = node.metadata.copy()
            text = node.text.strip()
            
            # 如果是标题节点
            if 'heading_level' in metadata:
                heading_level = metadata['heading_level']
                # 更新当前标题层级
                current_headings[heading_level] = text
                # 清除更深层级的标题
                keys_to_remove = [k for k in current_headings.keys() if k > heading_level]
                for k in keys_to_remove:
                    del current_headings[k]
                
                # 为标题节点添加标题路径
                heading_path = []
                for level in sorted(current_headings.keys()):
                    heading_path.append(current_headings[level])
                metadata['heading_path'] = heading_path
                metadata['current_heading'] = text
                metadata['is_heading'] = True
            else:
                # 为正文节点添加对应的最小级别标题
                if current_headings:
                    # 获取最深层级的标题作为当前段落的标题
                    max_level = max(current_headings.keys())
                    metadata['current_heading'] = current_headings[max_level]
                    metadata['heading_level'] = max_level
                    
                    # 添加完整的标题路径
                    heading_path = []
                    for level in sorted(current_headings.keys()):
                        heading_path.append(current_headings[level])
                    metadata['heading_path'] = heading_path
                else:
                    metadata['current_heading'] = None
                    metadata['heading_path'] = []
                
                metadata['is_heading'] = False
            
            documents.append({
                "content": text,
                "metadata": metadata
            })
        
        return documents

    def _process_markdown_file(self, file_path: str, file_content: str) -> List[Dict[str, Any]]:
        """处理 Markdown 文件，按标题+段落切分"""
        # 创建 Document 对象
        doc = Document(
            text=file_content,
            metadata={
                "source": file_path,
                "file_type": ".md"
            }
        )
        
        # 使用 MarkdownNodeParser 解析
        nodes = self.markdown_parser.get_nodes_from_documents([doc])
        
        # 提取标题上下文并返回
        return self._extract_heading_context(nodes)

    def load(self) -> List[Dict[str, Any]]:
        """加载文件并返回文档列表"""
        path = Path(self.path)
        if not path.exists():
            raise FileNotFoundError(f"文件或目录不存在: {self.path}")

        documents = []
        if path.is_file():
            # 处理单个文件
            file_ext = path.suffix.lower()
            if file_ext in self.file_types:
                if file_ext == ".pdf":
                    reader = PDFReader()
                    docs = reader.load_data(file_path=str(path))
                    for doc in docs:
                        documents.append({
                            "content": doc.text,
                            "metadata": {
                                "source": str(path),
                                "file_type": file_ext,
                            }
                        })
                elif file_ext == ".html":
                    reader = BeautifulSoupWebReader()
                    docs = reader.load_data(file_path=str(path))
                    for doc in docs:
                        documents.append({
                            "content": doc.text,
                            "metadata": {
                                "source": str(path),
                                "file_type": file_ext,
                            }
                        })
                elif file_ext == ".md" and self.enable_markdown_parsing:
                    # 特殊处理 Markdown 文件
                    with open(path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    documents.extend(self._process_markdown_file(str(path), content))
                else:  # .txt 和其他文本文件，或禁用 Markdown 解析的 .md 文件
                    reader = SimpleDirectoryReader(input_files=[str(path)])
                    docs = reader.load_data()
                    for doc in docs:
                        documents.append({
                            "content": doc.text,
                            "metadata": {
                                "source": str(path),
                                "file_type": file_ext,
                            }
                        })
        else:  # 处理目录
            # 获取目录下所有符合条件的文件
            all_files = []
            for file_type in self.file_types:
                all_files.extend(list(path.glob(f"**/*{file_type}")))
            
            if not all_files:
                return documents
            
            # 分别处理不同类型的文件
            for file_path in all_files:
                file_ext = file_path.suffix.lower()
                
                if file_ext == ".md" and self.enable_markdown_parsing:
                    # 特殊处理 Markdown 文件
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    documents.extend(self._process_markdown_file(str(file_path), content))
                else:
                    # 其他文件类型使用原有逻辑
                    reader = SimpleDirectoryReader(input_files=[str(file_path)])
                    docs = reader.load_data()
                    for doc in docs:
                        source = doc.metadata.get("file_path", str(file_path))
                        documents.append({
                            "content": doc.text,
                            "metadata": {
                                "source": source,
                                "file_type": file_ext,
                            }
                        })
        
        return documents


class WebLoader(BaseLoader):
    """网页加载器，支持加载网页内容"""

    def __init__(self, urls: List[str]):
        """初始化网页加载器

        Args:
            urls: 网页URL列表
        """
        self.urls = urls

    def load(self) -> List[Dict[str, Any]]:
        """加载网页并返回文档列表"""
        if not self.urls:
            return []

        reader = SimpleWebPageReader(html_to_text=True)
        docs = reader.load_data(urls=self.urls)

        documents = []
        for doc in docs:
            documents.append({
                "content": doc.text,
                "metadata": {
                    "source": doc.metadata.get("url", ""),
                    "file_type": "html",
                }
            })

        return documents