from utils.llm import LLMFactory
from utils.logger import get_logger
from utils.rag.index.factory import create_index
from utils.rag.loader.file_loader import FileLoader
from utils.rag.query import create_query_expansion
from utils.rag.reranker import create_reranker
from utils.rag.retriever import create_retriever
from utils.rag.retriever.bm25_retriever import BM25Retriever
from utils.rag.retriever.embedding_retriever import EmbeddingRetriever
from utils.rag.retriever.hybrid_retriever import HybridRetriever
from utils.rag.splitter import create_splitter
from utils.rag.embedding.factory import EmbeddingFactory
from utils.config import config

from .base_pipeline import BasePipeline


class AdvancedRagPipeline(BasePipeline):
    def __init__(self, pipeline_config: dict):
        super().__init__()
        self.pipeline_config = pipeline_config
        self.loader_cfg = pipeline_config["loader"]
        self.splitter_cfg = pipeline_config["splitter"]
        self.index_cfg = pipeline_config["index"]
        self.retriever_cfg = pipeline_config["retriever"]
        self.reranker_cfg = pipeline_config.get("reranker")
        self.query_expansion_cfg = pipeline_config.get("query_expansion")
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
        if self.retriever_cfg["type"] == "hybrid":
            self.index = {}
            retriever_objs = []
            for retriever_cfg in self.retriever_cfg["retrievers"]:
                idx_type = retriever_cfg["type"]
                if idx_type == "embedding":
                    emb_cfg = config.embeddings.clients.hf
                    embedding_client = EmbeddingFactory.create("hf", model_name=emb_cfg.model_name)
                    persist_dir = config.index.clients.embedding["persist_dir"]
                    idx = create_index(idx_type, embedding_client=embedding_client)
                    build_ok = idx.build(self.nodes, save_path=persist_dir)
                else:
                    idx = create_index(idx_type)
                    build_ok = idx.build(self.nodes)
                if not build_ok:
                    self.logger.error(f"[Pipeline] {idx_type} 索引构建失败！")
                    raise RuntimeError(f"{idx_type} 索引构建失败！")
                self.index[idx_type] = idx
                if idx_type == "embedding":
                    obj = EmbeddingRetriever(
                        **{k: v for k, v in retriever_cfg.items() if k != "type"}
                    )
                    obj.index_type = "embedding"
                elif idx_type == "bm25":
                    obj = BM25Retriever(
                        **{k: v for k, v in retriever_cfg.items() if k != "type"}
                    )
                    obj.index_type = "bm25"
                else:
                    raise ValueError(f"暂不支持的retriever类型: {idx_type}")
                retriever_objs.append(obj)
            self.retriever_obj = HybridRetriever(
                strategy=self.retriever_cfg.get("strategy", "weighted"),
                normalization=self.retriever_cfg.get("normalization", "minmax"),
                weights=self.retriever_cfg.get("weights"),
                retriever_objs=retriever_objs,
                top_k=self.retriever_cfg.get("top_k", 5),
            )
            self.logger.info("[Pipeline] embedding/bm25索引构建完成")
        else:
            if self.index_cfg["type"] == "embedding":
                emb_cfg = config.embeddings.clients.hf
                embedding_client = EmbeddingFactory.create("hf", model_name=emb_cfg.model_name)
                persist_dir = config.index.clients.embedding["persist_dir"]
                index = create_index(self.index_cfg["type"], embedding_client=embedding_client)
                build_ok = index.build(self.nodes, save_path=persist_dir)
            else:
                index = create_index(self.index_cfg["type"])
                build_ok = index.build(self.nodes)
            if not build_ok:
                self.logger.error("[Pipeline] 索引构建失败！")
                raise RuntimeError("索引构建失败！")
            self.index = index
            self.logger.info("[Pipeline] 索引构建完成")
            self.retriever_obj = create_retriever(
                self.retriever_cfg["type"], self.retriever_cfg
            )

    def run(self, query: str):
        self.logger.info("[Pipeline] 开始执行 Advanced RAG Pipeline (query only)")
        if self.index is None or self.nodes is None:
            raise RuntimeError("Pipeline未初始化，请先调用prepare()！")
        # 1. query expansion
        queries = [query]
        if self.query_expansion_cfg:
            self.logger.info("[Pipeline] 执行Query Expansion...")
            expander = create_query_expansion(
                self.query_expansion_cfg["type"], self.query_expansion_cfg
            )
            queries = expander.transform(query)
            if isinstance(queries, str):
                queries = [queries]
            self.logger.info(f"[Pipeline] Query Expansion生成 {len(queries)} 个查询")
        self.logger.info(f"[Pipeline] 查询: {queries}")
        # 2. 检索
        self.logger.info("[Pipeline] 检索相关内容...")
        retrieved_nodes = []
        if self.retriever_cfg["type"] == "hybrid":
            # 统计各子retriever召回数量
            for q in queries:
                all_sub_results = []
                # 只有HybridRetriever有retriever_objs
                for i, retriever in enumerate(getattr(self.retriever_obj, "retriever_objs", [])):
                    idx_type = getattr(retriever, "index_type", None)
                    sub_index = self.index[idx_type] if isinstance(self.index, dict) else self.index
                    sub_results = retriever.retrieve(sub_index, q)
                    all_sub_results.append((idx_type, len(sub_results)))
                for idx_type, count in all_sub_results:
                    self.logger.info(f"[Pipeline] {idx_type}召回: {count}")
                # 融合检索
                retrieved_nodes.extend(self.retriever_obj.retrieve(self.index, q))
        else:
            for q in queries:
                retrieved_nodes.extend(self.retriever_obj.retrieve(self.index, q))
        self.logger.info(f"[Pipeline] 检索到 {len(retrieved_nodes)} 个相关Node")
        # 3. rerank
        if self.reranker_cfg:
            self.logger.info("[Pipeline] 执行Rerank...")
            reranker = create_reranker(self.reranker_cfg["type"])
            retrieved_nodes = reranker.rerank(query, retrieved_nodes)
            self.logger.info("[Pipeline] Rerank完成")
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
