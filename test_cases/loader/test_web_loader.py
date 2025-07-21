from unittest.mock import patch, MagicMock
from utils.rag.loader.web_loader import WebLoader
from llama_index.core.schema import Document

HTML_SAMPLE = "<html><body><p>This is a test web page.</p></body></html>"
TEXT_SAMPLE = "This is a test web page."


@patch("llama_index.readers.web.simple_web.base.requests.get")
def test_load_web_page_text(mock_get):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.text = HTML_SAMPLE
    mock_get.return_value = mock_response

    loader = WebLoader(urls=["https://example.com"], html_to_text=True)
    documents = loader.load()

    assert isinstance(documents, list)
    assert len(documents) == 1
    assert isinstance(documents[0], Document)
    assert TEXT_SAMPLE in documents[0].text
