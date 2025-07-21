import os
from utils.rag.loader.file_loader import FileLoader
from utils.rag.splitter import HierarchicalSplitter
from utils.rag.index import create_index, EmbeddingIndex
from utils.rag.retriever import create_retriever
from utils.config import config
from utils.rag.pipeline import create_pipeline

def add_metadata_to_nodes(nodes, file_path):
    # 假设每个node有text属性，可以根据内容简单打标签
    for node in nodes:
        # 这里可以根据实际内容做更复杂的metadata提取
        node.metadata = node.metadata or {}
        node.metadata["source_file"] = os.path.basename(file_path)
        if "安装" in node.text:
            node.metadata["section"] = "安装"
        elif "命令" in node.text:
            node.metadata["section"] = "命令"
        elif "总结" in node.text:
            node.metadata["section"] = "总结"
        else:
            node.metadata["section"] = "其它"
    return nodes

def main():
    # 用pipeline方式
    pipeline = create_pipeline("naive_rag")
    query = "如何安装Perf工具？"
    result = pipeline.run(query)
    print("=== Pipeline 检索结果 ===")
    print(result)

if __name__ == "__main__":
    main()

