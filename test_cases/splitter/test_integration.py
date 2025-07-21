import pytest
import tempfile
import os
from pathlib import Path
from utils.rag.loader.file_loader import FileLoader
from utils.rag.splitter import create_splitter
from utils.rag.splitter.text_splitter import TextSplitter


class TestSplitterIntegration:
    """Splitter与Loader集成测试"""
    
    @pytest.fixture
    def temp_docs_dir(self):
        """创建临时文档目录"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # 创建测试文档
            doc1_path = Path(temp_dir) / "doc1.txt"
            doc1_path.write_text("""
            Linux性能分析工具Perf简介
            
            介绍
            Perf是一个基于Linux 2.6 +系统的分析工具，它抽象了在Linux中性能度量中CPU的硬件差异，提供一个简单的命令行界面。
            
            Perf基于最新版本Linux内核的perf_events接口。这篇文章通过示例展示了Perf工具的使用。
            
            命令
            Perf工具提供了一组丰富的命令来收集和分析性能和跟踪数据。命令行的用法与git类似，通过一个通用的命令Perf，实现了一组子命令。
            """)
            
            doc2_path = Path(temp_dir) / "doc2.txt"
            doc2_path.write_text("""
            性能分析基础
            
            性能分析是系统优化的重要步骤。通过分析系统的性能瓶颈，我们可以找到优化的方向。
            
            常见的性能指标包括：
            1. CPU使用率
            2. 内存使用率
            3. 磁盘I/O
            4. 网络I/O
            
            这些指标帮助我们了解系统的整体性能状况。
            """)
            
            yield temp_dir
    
    def test_loader_with_text_splitter(self, temp_docs_dir):
        """测试FileLoader与TextSplitter的集成"""
        # 加载文档
        loader = FileLoader(path=temp_docs_dir, file_types=[".txt"])
        documents = loader.load()
        
        assert len(documents) == 2
        
        # 创建分割器
        splitter = TextSplitter(
            chunk_size=200,
            chunk_overlap=50,
            split_method="char"
        )
        
        # 分割文档
        chunks = splitter.split(documents)
        
        # 验证结果
        assert len(chunks) > len(documents)  # 应该有更多的chunks
        
        # 检查每个chunk的metadata
        for chunk in chunks:
            assert "chunk_id" in chunk.metadata
            assert "total_chunks" in chunk.metadata
            assert "chunk_size" in chunk.metadata
            assert "split_method" in chunk.metadata
            assert chunk.metadata["split_method"] == "char"
            
            # 检查chunk大小
            assert len(chunk.text) <= 200
        
        # 验证所有chunks的文本长度合理
        total_text_length = sum(len(chunk.text) for chunk in chunks)
        original_text_length = sum(len(doc.text) for doc in documents)
        
        # 由于重叠，分割后的总长度可能大于原长度
        assert total_text_length >= original_text_length * 0.8
    
    def test_loader_with_recursive_splitter(self, temp_docs_dir):
        """测试FileLoader与RecursiveSplitter的集成"""
        from utils.rag.splitter.recursive_splitter import RecursiveSplitter
        
        # 加载文档
        loader = FileLoader(path=temp_docs_dir, file_types=[".txt"])
        documents = loader.load()
        
        # 创建递归分割器
        splitter = RecursiveSplitter(
            chunk_size=300,
            chunk_overlap=50,
            separators=["\n\n", "\n", "。", "！", "？"]
        )
        
        # 分割文档
        chunks = splitter.split(documents)
        
        # 验证结果
        assert len(chunks) > len(documents)
        
        for chunk in chunks:
            assert "split_method" in chunk.metadata
            assert chunk.metadata["split_method"] == "recursive"
            assert len(chunk.text) <= 300
    
    def test_loader_with_semantic_splitter(self, temp_docs_dir):
        """测试FileLoader与SemanticSplitter的集成"""
        from utils.rag.splitter.semantic_splitter import SemanticSplitter
        
        # 加载文档
        loader = FileLoader(path=temp_docs_dir, file_types=[".txt"])
        documents = loader.load()
        
        # 创建语义分割器（不使用embedding模型）
        splitter = SemanticSplitter(
            chunk_size=400,
            chunk_overlap=100,
            embedding_model=None  # 不使用embedding模型
        )
        
        # 分割文档
        chunks = splitter.split(documents)
        
        # 验证结果
        assert len(chunks) > len(documents)
        
        for chunk in chunks:
            assert "split_method" in chunk.metadata
            assert chunk.metadata["split_method"] == "semantic"
            assert len(chunk.text) <= 400
    
    def test_factory_with_loader(self, temp_docs_dir):
        """测试使用工厂模式创建splitter并与loader集成"""
        # 加载文档
        loader = FileLoader(path=temp_docs_dir, file_types=[".txt"])
        documents = loader.load()
        
        # 使用工厂创建分割器
        config = {
            "type": "text",
            "chunk_size": 150,
            "chunk_overlap": 30,
            "split_method": "char"
        }
        
        splitter = create_splitter(config)
        chunks = splitter.split(documents)
        
        # 验证结果
        assert len(chunks) > len(documents)
        assert isinstance(splitter, TextSplitter)
        
        for chunk in chunks:
            assert len(chunk.text) <= 150
    
    def test_empty_documents(self, temp_docs_dir):
        """测试空文档的处理"""
        # 创建空文档
        empty_doc_path = Path(temp_docs_dir) / "empty.txt"
        empty_doc_path.write_text("")
        
        # 加载文档
        loader = FileLoader(path=temp_docs_dir, file_types=[".txt"])
        documents = loader.load()
        
        # 创建分割器
        splitter = TextSplitter(chunk_size=100, split_method="char")
        chunks = splitter.split(documents)
        
        # 验证结果
        assert len(chunks) >= len(documents)  # 至少应该有相同数量的chunks
        
        # 检查空文档的处理
        empty_chunks = [chunk for chunk in chunks if not chunk.text.strip()]
        assert len(empty_chunks) > 0  # 应该有处理空文档的chunks
    
    def test_large_documents(self, temp_docs_dir):
        """测试大文档的处理"""
        # 创建大文档
        large_doc_path = Path(temp_docs_dir) / "large.txt"
        large_text = "这是一个测试句子。" * 1000  # 创建大文档
        large_doc_path.write_text(large_text)
        
        # 加载文档
        loader = FileLoader(path=temp_docs_dir, file_types=[".txt"])
        documents = loader.load()
        
        # 创建分割器
        splitter = TextSplitter(chunk_size=100, chunk_overlap=20, split_method="char")
        chunks = splitter.split(documents)
        
        # 验证结果
        assert len(chunks) > len(documents)
        
        # 检查chunk大小
        for chunk in chunks:
            assert len(chunk.text) <= 100
        
        # 验证所有文本都被处理
        original_text = "".join(doc.text for doc in documents)
        processed_text = "".join(chunk.text for chunk in chunks)
        
        # 由于重叠，处理后的文本可能包含重复内容，但应该包含所有原始内容
        for char in original_text:
            assert char in processed_text 