import logging
from utils.rag.loader.file_loader import FileLoader
from utils.rag.splitter import create_splitter
from utils.rag.index import create_index, EmbeddingIndex, BM25Index
from utils.rag.retriever import create_retriever
from utils.config import config
from utils.llm import LLMFactory

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # 1. 加载文档
    data_dir = "./test_cases/data"
    loader = FileLoader(path=data_dir, file_types=[".md", ".txt", ".html"])
    documents = loader.load()
    logger.info(f"加载了 {len(documents)} 个文档")

    # 2. 分割文档
    splitter = create_splitter(config.splitter.hierarchical)
    nodes = splitter.split(documents)
    logger.info(f"分割得到 {len(nodes)} 个结构化Node")

    # 3. 构建索引
    index_type = config.index.clients[config.index.default]["type"]
    retriever_cfg = config.retriever.clients[config.retriever.default]
    retriever_type = retriever_cfg["type"]
    index = create_index(index_type)
    build_ok = index.build(nodes)
    if not build_ok:
        logger.error("索引构建失败！")
        return
    logger.info("索引构建完成")
    save_ok = index.save(config.index.clients[config.index.default]["persist_dir"])
    if not save_ok:
        logger.error("索引保存失败！")
        return
    logger.info(f"索引已保存到: {config.index.clients[config.index.default]['persist_dir']}")

    # 4. 检索+LLM分析
    query = "Linux性能分析工具Perf的主要特性有哪些？"
    logger.info(f"用户问题: {query}")
    retriever = create_retriever(retriever_type)
    top_k = retriever_cfg.get("top_k", 5)
    retrieved_nodes = retriever.retrieve(index, query, top_k=top_k)
    logger.info(f"检索到 {len(retrieved_nodes)} 个相关Node")
    context = "\n\n".join([n.get_content() for n in retrieved_nodes])

    # 5. 用自定义LLM分析
    llm = LLMFactory.from_config()
    prompt = f"已知信息如下：\n{context}\n\n请根据上述内容回答：{query}"
    answer = llm.completion(prompt)
    logger.info(f"LLM生成答案: {answer}")
    print("\n=== 最终答案 ===\n", answer)

if __name__ == "__main__":
    main() 