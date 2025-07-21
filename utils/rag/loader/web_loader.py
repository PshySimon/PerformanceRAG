from llama_index.readers.web import SimpleWebPageReader
from llama_index.core.schema import Document
from .base_loader import BaseLoader


class WebLoader(BaseLoader):
    def __init__(self, urls: list[str], html_to_text: bool = False):
        self.urls = urls
        self.html_to_text = html_to_text

    def load(self) -> list[Document]:
        reader = SimpleWebPageReader(html_to_text=self.html_to_text)
        return reader.load_data(self.urls)
