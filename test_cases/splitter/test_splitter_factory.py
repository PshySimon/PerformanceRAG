import pytest
from utils.rag.splitter.splitter_factory import create_splitter
from utils.rag.splitter.text_splitter import TextSplitter
from utils.rag.splitter.recursive_splitter import RecursiveSplitter
from utils.rag.splitter.semantic_splitter import SemanticSplitter


class TestSplitterFactory:
    """SplitterFactory测试类"""
    
    def test_create_text_splitter(self):
        """测试创建TextSplitter"""
        config = {
            "type": "text",
            "chunk_size": 1000,
            "chunk_overlap": 200,
            "split_method": "char",
            "separator": "\n",
            "keep_separator": True
        }
        
        splitter = create_splitter(config)
        assert isinstance(splitter, TextSplitter)
        assert splitter.chunk_size == 1000
        assert splitter.chunk_overlap == 200
        assert splitter.split_method == "char"
    
    def test_create_recursive_splitter(self):
        """测试创建RecursiveSplitter"""
        config = {
            "type": "recursive",
            "chunk_size": 1000,
            "chunk_overlap": 200,
            "separators": ["\n\n", "\n", "。", "！", "？"],
            "keep_separator": True
        }
        
        splitter = create_splitter(config)
        assert isinstance(splitter, RecursiveSplitter)
        assert splitter.chunk_size == 1000
        assert splitter.chunk_overlap == 200
        assert len(splitter.separators) == 5
    
    def test_create_semantic_splitter(self):
        """测试创建SemanticSplitter"""
        config = {
            "type": "semantic",
            "chunk_size": 1000,
            "chunk_overlap": 200,
            "embedding_model": "text-embedding-ada-002",
            "similarity_threshold": 0.8,
            "min_chunk_size": 100
        }
        
        splitter = create_splitter(config)
        assert isinstance(splitter, SemanticSplitter)
        assert splitter.chunk_size == 1000
        assert splitter.chunk_overlap == 200
        assert splitter.embedding_model == "text-embedding-ada-002"
        assert splitter.similarity_threshold == 0.8
    
    def test_create_with_defaults(self):
        """测试使用默认值创建splitter"""
        config = {"type": "text"}
        
        splitter = create_splitter(config)
        assert isinstance(splitter, TextSplitter)
        assert splitter.chunk_size == 1000  # 默认值
        assert splitter.chunk_overlap == 200  # 默认值
    
    def test_create_invalid_type(self):
        """测试创建无效类型的splitter"""
        config = {"type": "invalid"}
        
        with pytest.raises(ValueError, match="不支持的Splitter类型"):
            create_splitter(config)
    
    def test_create_without_type(self):
        """测试没有指定类型时使用默认值"""
        config = {}
        
        splitter = create_splitter(config)
        assert isinstance(splitter, TextSplitter)  # 默认类型
    
    def test_text_splitter_with_all_options(self):
        """测试TextSplitter的所有选项"""
        config = {
            "type": "text",
            "chunk_size": 500,
            "chunk_overlap": 100,
            "split_method": "word",
            "separator": " ",
            "keep_separator": False
        }
        
        splitter = create_splitter(config)
        assert isinstance(splitter, TextSplitter)
        assert splitter.chunk_size == 500
        assert splitter.chunk_overlap == 100
        assert splitter.split_method == "word"
        assert splitter.separator == " "
        assert splitter.keep_separator == False
    
    def test_recursive_splitter_with_custom_separators(self):
        """测试RecursiveSplitter的自定义分隔符"""
        custom_separators = ["\n\n", "\n", ".", "!", "?"]
        config = {
            "type": "recursive",
            "separators": custom_separators
        }
        
        splitter = create_splitter(config)
        assert isinstance(splitter, RecursiveSplitter)
        assert splitter.separators == custom_separators
    
    def test_semantic_splitter_without_embedding_model(self):
        """测试SemanticSplitter没有embedding模型的情况"""
        config = {
            "type": "semantic",
            "chunk_size": 800,
            "similarity_threshold": 0.7
        }
        
        splitter = create_splitter(config)
        assert isinstance(splitter, SemanticSplitter)
        assert splitter.embedding_model is None
        assert splitter.similarity_threshold == 0.7 