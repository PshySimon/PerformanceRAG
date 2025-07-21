from pathlib import Path
from utils.logger import get_logger
from llama_index.core import SimpleDirectoryReader
from llama_index.readers.file import PDFReader
from llama_index.core import Document
from .base_loader import BaseLoader
from typing import Optional

# 新增导入
try:
    from llama_index.readers.web import BeautifulSoupWebReader
    HAS_BS_READER = True
except ImportError:
    HAS_BS_READER = False

class FileLoader(BaseLoader):
    def __init__(self, path: str, file_types: Optional[list[str]] = None):
        self.path = Path(path)
        self.logger = get_logger(__name__)
        self.file_types = file_types or ["txt", "md", "pdf", "html"]

    def load(self) -> list[Document]:
        documents = []
        self.logger.info(f"从{self.path}加载文件中...")

        if not self.path.exists():
            raise FileNotFoundError(f"路径不存在: {self.path}")

        if self.path.is_file():
            ext = self.path.suffix.lower().lstrip(".")
            if ext == "pdf":
                reader = PDFReader()
                documents = reader.load_data(file=self.path)
            elif ext == "html":
                if HAS_BS_READER:
                    reader = BeautifulSoupWebReader()
                    documents = reader.load_data([str(self.path)])
                else:
                    raise ImportError("需要安装 llama-index[web] 支持 HTML 结构化解析")
            else:
                with open(self.path, "r", encoding="utf-8") as f:
                    text = f.read()
                documents = [Document(text=text)]
        else:
            reader = SimpleDirectoryReader(
                input_dir=str(self.path),
                required_exts=self.file_types,
                recursive=True
            )
            documents = reader.load_data()

        return documents
