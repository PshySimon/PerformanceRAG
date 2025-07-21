import logging
from utils.rag.pipeline import create_pipeline


def main():
    logging.basicConfig(level=logging.INFO)

    pipeline = create_pipeline("advanced_rag")
    query = "Linux性能分析工具Perf的主要特性有哪些？"
    result = pipeline.run(query)
    print("=== Advanced RAG Pipeline 结果 ===")
    print(result)


if __name__ == "__main__":
    main()
