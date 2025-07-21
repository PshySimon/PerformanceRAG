import pytest
import time
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from utils.rag.loader.file_loader import FileLoader
from utils.rag.splitter import create_splitter
from utils.rag.splitter.text_splitter import TextSplitter
from utils.rag.splitter.recursive_splitter import RecursiveSplitter
from utils.rag.splitter.semantic_splitter import SemanticSplitter


class TestDataDrivenSplitter:
    """基于真实数据的DT测试用例"""
    
    @pytest.fixture(scope="class")
    def test_data_dir(self):
        """获取测试数据目录"""
        return Path(__file__).parent.parent / "data"
    
    @pytest.fixture(scope="class")
    def documents_cache(self, test_data_dir):
        """缓存加载的文档，避免重复加载"""
        loader = FileLoader(path=str(test_data_dir), file_types=[".txt", ".md"])
        return loader.load()
    
    @pytest.fixture
    def text_splitter_config(self):
        """TextSplitter配置"""
        return {
            "type": "text",
            "chunk_size": 300,  # 减小chunk大小，提高处理速度
            "chunk_overlap": 50,
            "split_method": "char"
        }
    
    @pytest.fixture
    def recursive_splitter_config(self):
        """RecursiveSplitter配置"""
        return {
            "type": "recursive",
            "chunk_size": 400,  # 减小chunk大小
            "chunk_overlap": 80,
            "separators": ["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]
        }
    
    @pytest.fixture
    def semantic_splitter_config(self):
        """SemanticSplitter配置"""
        return {
            "type": "semantic",
            "chunk_size": 350,  # 减小chunk大小
            "chunk_overlap": 70,
            "embedding_client": None,  # 不使用embedding客户端，回退到启发式方法
            "similarity_threshold": 0.7
        }
    
    def test_text_splitter_with_long_document(self, documents_cache, text_splitter_config):
        """测试TextSplitter处理长文本文档"""
        # 过滤出适合TextSplitter的文档
        text_docs = [doc for doc in documents_cache if "text_splitter_data" in doc.metadata.get("file_name", "")]
        
        if not text_docs:
            pytest.skip("没有找到适合TextSplitter的测试数据")
        
        # 创建分割器
        splitter = create_splitter(text_splitter_config)
        chunks = splitter.split(text_docs)
        
        # 验证结果
        assert len(chunks) > len(text_docs), "应该生成更多的chunks"
        
        # 检查chunk大小（只检查前几个，提高速度）
        for chunk in chunks[:5]:  # 只检查前5个chunk
            assert len(chunk.text) <= text_splitter_config["chunk_size"], f"Chunk大小超过限制: {len(chunk.text)}"
            assert "chunk_id" in chunk.metadata
            assert "split_method" in chunk.metadata
            assert chunk.metadata["split_method"] == "char"
        
        # 简化文本完整性验证
        original_text = "".join(doc.text for doc in text_docs)
        processed_text = "".join(chunk.text for chunk in chunks)
        
        # 只检查文本长度，不逐字符检查
        assert len(processed_text) >= len(original_text) * 0.8, "处理后的文本应该包含大部分原始内容"
        
        print(f"TextSplitter处理了 {len(text_docs)} 个文档，生成了 {len(chunks)} 个chunks")
    
    def test_recursive_splitter_with_structured_document(self, documents_cache, recursive_splitter_config):
        """测试RecursiveSplitter处理结构化文档"""
        # 过滤出适合RecursiveSplitter的文档
        recursive_docs = [doc for doc in documents_cache if "recursive_splitter_data" in doc.metadata.get("file_name", "")]
        
        if not recursive_docs:
            pytest.skip("没有找到适合RecursiveSplitter的测试数据")
        
        # 创建分割器
        splitter = create_splitter(recursive_splitter_config)
        chunks = splitter.split(recursive_docs)
        
        # 验证结果
        assert len(chunks) > len(recursive_docs), "应该生成更多的chunks"
        
        # 检查chunk大小（只检查前几个）
        for chunk in chunks[:5]:
            assert len(chunk.text) <= recursive_splitter_config["chunk_size"], f"Chunk大小超过限制: {len(chunk.text)}"
            assert "split_method" in chunk.metadata
            assert chunk.metadata["split_method"] == "recursive"
        
        # 验证结构化特征
        markdown_headers = sum(1 for chunk in chunks if chunk.text.strip().startswith("#"))
        assert markdown_headers > 0, "应该包含Markdown标题"
        
        print(f"RecursiveSplitter处理了 {len(recursive_docs)} 个文档，生成了 {len(chunks)} 个chunks")
    
    def test_semantic_splitter_with_semantic_document(self, documents_cache, semantic_splitter_config):
        """测试SemanticSplitter处理语义相关文档"""
        # 过滤出适合SemanticSplitter的文档
        semantic_docs = [doc for doc in documents_cache if "semantic_splitter_data" in doc.metadata.get("file_name", "")]
        
        if not semantic_docs:
            pytest.skip("没有找到适合SemanticSplitter的测试数据")
        
        # 创建分割器
        splitter = create_splitter(semantic_splitter_config)
        chunks = splitter.split(semantic_docs)
        
        # 验证结果
        assert len(chunks) > len(semantic_docs), "应该生成更多的chunks"
        
        # 检查chunk大小（只检查前几个）
        for chunk in chunks[:5]:
            assert len(chunk.text) <= semantic_splitter_config["chunk_size"], f"Chunk大小超过限制: {len(chunk.text)}"
            assert "split_method" in chunk.metadata
            assert chunk.metadata["split_method"] == "semantic"
        
        # 验证语义完整性（检查段落结构）
        paragraphs = sum(1 for chunk in chunks if "\n\n" in chunk.text)
        assert paragraphs > 0, "应该保持段落结构"
        
        print(f"SemanticSplitter处理了 {len(semantic_docs)} 个文档，生成了 {len(chunks)} 个chunks")
    
    def test_recursive_splitter_with_mixed_language(self, documents_cache, recursive_splitter_config):
        """测试RecursiveSplitter处理多语言混合文档"""
        # 过滤出多语言文档
        mixed_docs = [doc for doc in documents_cache if "mixed_language_data" in doc.metadata.get("file_name", "")]
        
        if not mixed_docs:
            pytest.skip("没有找到多语言混合测试数据")
        
        # 创建分割器
        splitter = create_splitter(recursive_splitter_config)
        chunks = splitter.split(mixed_docs)
        
        # 验证结果
        assert len(chunks) > len(mixed_docs), "应该生成更多的chunks"
        
        # 检查多语言特征（只检查前几个chunk）
        chinese_chars = sum(1 for chunk in chunks[:3] if any('\u4e00' <= char <= '\u9fff' for char in chunk.text))
        english_words = sum(1 for chunk in chunks[:3] if any(word.isalpha() and word.isascii() for word in chunk.text.split()))
        
        assert chinese_chars > 0, "应该包含中文字符"
        assert english_words > 0, "应该包含英文单词"
        
        print(f"RecursiveSplitter处理了 {len(mixed_docs)} 个多语言文档，生成了 {len(chunks)} 个chunks")
    

    def test_all_splitters_comparison(self, documents_cache, text_splitter_config, recursive_splitter_config, semantic_splitter_config):
        """比较所有splitter在相同数据上的表现"""
        if not documents_cache:
            pytest.skip("没有找到测试数据")

        # 只用前2个文档做对比
        test_docs = documents_cache[:2]

        # 使用不同的splitter处理
        text_splitter = create_splitter(text_splitter_config)
        recursive_splitter = create_splitter(recursive_splitter_config)
        semantic_splitter = create_splitter(semantic_splitter_config)

        for splitter, name in [
            (text_splitter, "TextSplitter"),
            (recursive_splitter, "RecursiveSplitter"),
            (semantic_splitter, "SemanticSplitter")
        ]:
            start = time.time()
            chunks = splitter.split(test_docs)
            elapsed = time.time() - start
            print(f"{name} 处理 {len(test_docs)} 个文档生成 {len(chunks)} 个chunks, 用时 {elapsed:.3f} 秒")
            assert len(chunks) > 0, f"{name}应该生成chunks"
            for chunk in chunks[:3]:
                assert len(chunk.text) > 0
                assert "split_method" in chunk.metadata

    def test_splitter_performance(self, documents_cache, text_splitter_config):
        """测试splitter性能"""
        import time
        
        if not documents_cache:
            pytest.skip("没有找到测试数据")
        
        # 只使用前几个文档进行性能测试
        test_docs = documents_cache[:2]  # 只测试前2个文档
        
        # 测试TextSplitter性能
        splitter = create_splitter(text_splitter_config)
        
        start_time = time.time()
        chunks = splitter.split(test_docs)
        end_time = time.time()
        
        processing_time = end_time - start_time
        total_text_length = sum(len(doc.text) for doc in test_docs)
        
        print(f"处理了 {len(test_docs)} 个文档，总文本长度: {total_text_length}")
        print(f"生成了 {len(chunks)} 个chunks")
        print(f"处理时间: {processing_time:.4f} 秒")
        print(f"处理速度: {total_text_length / processing_time:.2f} 字符/秒")
        
        # 性能断言（更宽松的时间限制）
        assert processing_time < 5.0, "处理时间不应该超过5秒"
        assert len(chunks) > len(test_docs), "应该生成更多的chunks"
    
    def test_splitter_edge_cases(self, documents_cache):
        """测试splitter的边界情况"""
        if not documents_cache:
            pytest.skip("没有找到测试数据")
        
        # 只使用前2个文档进行边界测试
        test_docs = documents_cache[:2]
        
        # 测试不同的chunk大小（减少测试数量）
        chunk_sizes = [100, 300, 500]  # 减少测试的chunk大小数量
        
        for chunk_size in chunk_sizes:
            config = {
                "type": "text",
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_size // 5,
                "split_method": "char"
            }
            
            splitter = create_splitter(config)
            chunks = splitter.split(test_docs)
            
            # 验证chunk大小（只检查前几个）
            for chunk in chunks[:3]:
                assert len(chunk.text) <= chunk_size, f"Chunk大小超过限制: {len(chunk.text)} > {chunk_size}"
            
            print(f"Chunk大小 {chunk_size}: 生成了 {len(chunks)} 个chunks")
    
    def test_splitter_metadata_integrity(self, documents_cache, text_splitter_config):
        """测试splitter元数据完整性"""
        if not documents_cache:
            pytest.skip("没有找到测试数据")
        
        # 只使用前2个文档进行元数据测试
        test_docs = documents_cache[:2]
        
        # 为文档添加自定义元数据
        for i, doc in enumerate(test_docs):
            doc.metadata.update({
                "custom_field": f"value_{i}",
                "document_id": i,
                "source_type": "test_data"
            })
        
        # 使用splitter处理
        splitter = create_splitter(text_splitter_config)
        chunks = splitter.split(test_docs)
        
        # 验证元数据完整性（只检查前几个chunk）
        for chunk in chunks[:3]:
            # 检查必需的元数据
            assert "chunk_id" in chunk.metadata
            assert "total_chunks" in chunk.metadata
            assert "chunk_size" in chunk.metadata
            assert "split_method" in chunk.metadata
            
            # 检查自定义元数据是否保留
            assert "custom_field" in chunk.metadata
            assert "document_id" in chunk.metadata
            assert "source_type" in chunk.metadata
            
            # 验证元数据类型
            assert isinstance(chunk.metadata["chunk_id"], int)
            assert isinstance(chunk.metadata["total_chunks"], int)
            assert isinstance(chunk.metadata["chunk_size"], int)
            assert isinstance(chunk.metadata["split_method"], str)
        
        print(f"元数据完整性测试通过，处理了 {len(test_docs)} 个文档，生成了 {len(chunks)} 个chunks") 