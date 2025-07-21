from utils.config import config
from utils.llm import LLMFactory
from utils.logger import get_logger
from utils.rag.embedding.factory import EmbeddingFactory
from utils.rag.index import create_index
from utils.rag.loader.file_loader import FileLoader
from utils.rag.retriever import create_retriever
from utils.rag.splitter import create_splitter

from .base_pipeline import BasePipeline


class NaiveRagPipeline(BasePipeline):
    def __init__(self, pipeline_config: dict):
        super().__init__()
        self.pipeline_config = pipeline_config
        self.loader_cfg = pipeline_config["loader"]
        self.splitter_cfg = pipeline_config["splitter"]
        self.index_cfg = pipeline_config["index"]
        self.retriever_cfg = pipeline_config["retriever"]
        self.llm_cfg = pipeline_config.get("llm", {})
        self.logger = get_logger(__name__)
        self.index = None
        self.nodes = None
        self.documents = None

    def _do_prepare(self):
        self.logger.info("[Pipeline] 预处理：加载文档、分割、构建索引...")
        loader = FileLoader(
            path=self.loader_cfg["path"], file_types=self.loader_cfg["file_types"]
        )
        self.documents = loader.load()
        self.logger.info(f"[Pipeline] 加载了 {len(self.documents)} 个文档")
        splitter = create_splitter(self.splitter_cfg)
        self.nodes = splitter.split(self.documents)
        self.logger.info(f"[Pipeline] 分割得到 {len(self.nodes)} 个结构化Node")
        if self.index_cfg["type"] == "embedding":
            emb_cfg = config.embeddings.clients.hf
            embedding_client = EmbeddingFactory.create(
                "hf", model_name=emb_cfg.model_name
            )
            persist_dir = config.index.clients.embedding["persist_dir"]
            index = create_index(
                self.index_cfg["type"], embedding_client=embedding_client
            )
            build_ok = index.build(self.nodes, save_path=persist_dir)
        else:
            index = create_index(self.index_cfg["type"])
            build_ok = index.build(self.nodes)
        if not build_ok:
            self.logger.error("[Pipeline] 索引构建失败！")
            raise RuntimeError("索引构建失败！")
        self.index = index
        self.logger.info("[Pipeline] 索引构建完成")

    def run(self, query: str):
        self.logger.info("[Pipeline] 开始执行 Naive RAG Pipeline (query only)")
        if self.index is None or self.nodes is None:
            raise RuntimeError("Pipeline未初始化，请先调用prepare()！")
        # 1. 查询（此处为query passthrough，后续可扩展query expansion等）
        queries = [query]
        self.logger.info(f"[Pipeline] 查询: {queries}")
        # 2. 检索
        self.logger.info("[Pipeline] 检索相关内容...")
        retriever = create_retriever(self.retriever_cfg["type"], self.retriever_cfg)
        retrieved_nodes = []
        for q in queries:
            retrieved_nodes.extend(retriever.retrieve(self.index, q))
        self.logger.info(f"[Pipeline] 检索到 {len(retrieved_nodes)} 个相关Node")
        # 3. rerank（naive无）
        # 4. LLM生成
        if self.llm_cfg.get("use", False):
            self.logger.info("[Pipeline] 调用LLM生成答案...")
            llm = LLMFactory.from_config()
            context = "\n\n".join([n.get_content() for n in retrieved_nodes])
            prompt = f"已知信息如下：\n{context}\n\n请根据上述内容回答：{query}"
            answer = llm.completion(prompt)
            self.logger.info("[Pipeline] LLM生成答案完成")
            return answer
        else:
            return retrieved_nodes
