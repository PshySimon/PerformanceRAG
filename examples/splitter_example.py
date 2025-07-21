#!/usr/bin/env python3
"""
Splitter模块使用示例
演示如何使用不同的splitter来分割文档
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.rag.loader.file_loader import FileLoader
from utils.rag.splitter import create_splitter
from utils.config import config


def main():
    """主函数，演示splitter的使用"""
    
    # 1. 加载文档
    print("=== 加载文档 ===")
    loader = FileLoader(
        path="./test_cases/data/",
        file_types=[".txt", ".md"]
    )
    documents = loader.load()
    print(f"加载了 {len(documents)} 个文档")
    
    if not documents:
        print("没有找到文档，请确保 ./test_cases/data/ 目录下有文档文件")
        return
    
    # 2. 使用不同的splitter进行分割
    print("\n=== 使用TextSplitter分割 ===")
    text_splitter_config = config.splitter.hierarchical
    text_splitter = create_splitter(text_splitter_config)
    text_chunks = text_splitter.split(documents)
    print(f"TextSplitter 生成了 {len(text_chunks)} 个chunks")
    
    # 显示前几个chunks
    for i, chunk in enumerate(text_chunks[:3]):
        print("-"*100)
        print(f"Chunk {i+1}: {chunk.text}...")
        print(f"Metadata: {chunk.metadata}")
        print("-"*100)
    
    print("\n=== 使用RecursiveSplitter分割 ===")
    recursive_splitter_config = config.splitter.recursive
    recursive_splitter = create_splitter(recursive_splitter_config)
    recursive_chunks = recursive_splitter.split(documents)
    print(f"RecursiveSplitter 生成了 {len(recursive_chunks)} 个chunks")
    
    # 显示前几个chunks
    for i, chunk in enumerate(recursive_chunks[:3]):
        print(f"Chunk {i+1}: {chunk.text[:100]}...")
        print(f"Metadata: {chunk.metadata}")
        print()
    
    print("\n=== 使用SemanticSplitter分割 ===")
    semantic_splitter_config = config.splitter.semantic
    semantic_splitter = create_splitter(semantic_splitter_config)
    semantic_chunks = semantic_splitter.split(documents)
    print(f"SemanticSplitter 生成了 {len(semantic_chunks)} 个chunks")
    
    # 显示前几个chunks
    for i, chunk in enumerate(semantic_chunks[:3]):
        print(f"Chunk {i+1}: {chunk.text[:100]}...")
        print(f"Metadata: {chunk.metadata}")
        print()


def test_single_document():
    """测试单个文档的分割"""
    from llama_index.core.schema import Document
    
    # 创建一个测试文档
    test_text = """
    Linux性能分析工具Perf简介
    
    介绍
    Perf是一个基于Linux 2.6 +系统的分析工具，它抽象了在Linux中性能度量中CPU的硬件差异，提供一个简单的命令行界面。
    
    Perf基于最新版本Linux内核的perf_events接口。这篇文章通过示例展示了Perf工具的使用。
    
    命令
    Perf工具提供了一组丰富的命令来收集和分析性能和跟踪数据。命令行的用法与git类似，通过一个通用的命令Perf，实现了一组子命令。
    """
    
    test_doc = Document(text=test_text, metadata={"source": "test.txt"})
    
    print("=== 测试单个文档分割 ===")
    
    # 使用TextSplitter
    text_splitter = create_splitter({
        "type": "text",
        "chunk_size": 200,
        "chunk_overlap": 50,
        "split_method": "char"
    })
    
    chunks = text_splitter.split([test_doc])
    print(f"生成了 {len(chunks)} 个chunks:")
    
    for i, chunk in enumerate(chunks):
        print(f"\nChunk {i+1}:")
        print(f"长度: {len(chunk.text)}")
        print(f"内容: {chunk.text}")
        print(f"Metadata: {chunk.metadata}")


if __name__ == "__main__":
    print("Splitter模块使用示例")
    print("=" * 50)
    
    # 测试单个文档
    test_single_document()
    
    print("\n" + "=" * 50)
    
    # 测试实际文档
    main() 