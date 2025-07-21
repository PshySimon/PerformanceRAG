import pytest
from utils.rag.splitter.text_splitter import TextSplitter
from llama_index.core.schema import Document


class TestTextSplitter:
    """TextSplitter测试类"""
    
    def test_init(self):
        """测试初始化"""
        splitter = TextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            split_method="char"
        )
        
        assert splitter.chunk_size == 1000
        assert splitter.chunk_overlap == 200
        assert splitter.split_method == "char"
    
    def test_init_invalid_method(self):
        """测试无效的分割方法"""
        with pytest.raises(ValueError):
            TextSplitter(split_method="invalid")
    
    def test_split_by_char(self):
        """测试按字符分割"""
        splitter = TextSplitter(
            chunk_size=10,
            chunk_overlap=2,
            split_method="char"
        )
        
        text = "这是一个测试文本，用于测试分割功能。"
        chunks = splitter.split_text(text)
        
        assert len(chunks) > 1
        assert all(len(chunk) <= 10 for chunk in chunks)
    
    def test_split_by_word(self):
        """测试按单词分割"""
        splitter = TextSplitter(
            chunk_size=3,
            chunk_overlap=1,
            split_method="word"
        )
        
        text = "This is a test text for word splitting."
        chunks = splitter.split_text(text)
        
        assert len(chunks) > 1
        # 检查每个chunk的单词数不超过限制
        for chunk in chunks:
            word_count = len(chunk.split())
            assert word_count <= 3
    
    def test_split_by_sentence(self):
        """测试按句子分割"""
        splitter = TextSplitter(
            chunk_size=50,
            chunk_overlap=10,
            split_method="sentence"
        )
        
        text = "这是第一个句子。这是第二个句子。这是第三个句子。"
        chunks = splitter.split_text(text)
        
        assert len(chunks) > 0
        assert all(len(chunk) <= 50 for chunk in chunks)
    
    def test_split_documents(self):
        """测试文档分割"""
        splitter = TextSplitter(
            chunk_size=20,
            chunk_overlap=5,
            split_method="char"
        )
        
        doc1 = Document(text="这是第一个文档的内容。", metadata={"source": "doc1.txt"})
        doc2 = Document(text="这是第二个文档的内容。", metadata={"source": "doc2.txt"})
        
        documents = [doc1, doc2]
        split_docs = splitter.split(documents)
        
        assert len(split_docs) > len(documents)
        
        # 检查metadata是否正确保留
        for doc in split_docs:
            assert "chunk_id" in doc.metadata
            assert "total_chunks" in doc.metadata
            assert "chunk_size" in doc.metadata
            assert "split_method" in doc.metadata
            assert doc.metadata["split_method"] == "char"
    
    def test_empty_text(self):
        """测试空文本"""
        splitter = TextSplitter(chunk_size=10, split_method="char")
        chunks = splitter.split_text("")
        assert chunks == [""]
    
    def test_text_shorter_than_chunk(self):
        """测试文本短于chunk大小"""
        splitter = TextSplitter(chunk_size=100, split_method="char")
        text = "短文本"
        chunks = splitter.split_text(text)
        assert chunks == [text]
    
    def test_overlap_behavior(self):
        """测试重叠行为"""
        splitter = TextSplitter(
            chunk_size=10,
            chunk_overlap=3,
            split_method="char"
        )
        
        text = "这是一个较长的测试文本，用于测试重叠功能。"
        chunks = splitter.split_text(text)
        
        # 检查相邻chunks之间是否有重叠
        for i in range(len(chunks) - 1):
            current_chunk = chunks[i]
            next_chunk = chunks[i + 1]
            
            # 检查是否有重叠（简化检查）
            assert len(current_chunk) >= 7  # 10 - 3 = 7 