from utils.rag.loader.file_loader import FileLoader
from utils.rag.splitter import HierarchicalSplitter


def main():
    data_dir = "./test_cases/data"
    loader = FileLoader(path=data_dir, file_types=[".md", ".html", ".txt"])
    documents = loader.load()
    print(f"加载了 {len(documents)} 个文档")

    splitter = HierarchicalSplitter()
    nodes = splitter.split(documents)
    print(f"分割得到 {len(nodes)} 个层次化Node")

    for i, node in enumerate(nodes[:10]):  # 只展示前10个
        print(f"--- Node {i+1} ---")
        print("内容:", node.text.strip()[:120].replace("\n", " "))
        print("元数据:", node.metadata)
        print()


if __name__ == "__main__":
    main() 