import pytest
from utils.rag.loader.file_loader import FileLoader


@pytest.fixture
def sample_txt_file(tmp_path):
    content = "This is a test text file."
    file = tmp_path / "sample.txt"
    file.write_text(content)
    return file


def test_load_txt_file(sample_txt_file):
    loader = FileLoader(path=str(sample_txt_file.parent), file_types=[".txt"])
    documents = loader.load()
    assert len(documents) == 1
    assert documents[0].text == "This is a test text file."
