from typing import Dict, Any, List
from .base_reranker import BaseRerankerComponent
# 修复导入路径
try:
    from rag.components.embedding.embedding_factory import EmbeddingFactory
except ImportError:
    # 如果上面的路径不存在，尝试其他可能的路径
    try:
        from rag.components.embedding.embedding_factory import EmbeddingFactory
    except ImportError:
        # 创建一个简单的工厂类作为备用
        class EmbeddingFactory:
            @staticmethod
            def create(embedding_type: str, **kwargs):
                if embedding_type == "hf":
                    from sentence_transformers import SentenceTransformer
                    model_name = kwargs.get("model_name", "sentence-transformers/all-MiniLM-L6-v2")
                    return SentenceTransformerWrapper(model_name)
                else:
                    raise ValueError(f"不支持的embedding类型: {embedding_type}")
        
        class SentenceTransformerWrapper:
            def __init__(self, model_name: str):
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer(model_name)
            
            def embed_query(self, query: str) -> List[float]:
                return self.model.encode([query])[0].tolist()
            
            def embed_documents(self, documents: List[str]) -> List[List[float]]:
                embeddings = self.model.encode(documents)
                return [emb.tolist() for emb in embeddings]

from utils.config import config
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class EmbeddingRerankerComponent(BaseRerankerComponent):
    """Embedding重排组件"""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        super().__init__(name, config)
        self.embedding_client = None
        self.similarity_metric = config.get("similarity_metric", "cosine")
        self.embedding_type = config.get("embedding_type", "hf")
        
    def _do_initialize(self):
        """初始化embedding客户端"""
        try:
            if self.embedding_type == "hf":
                emb_config = config.embeddings.clients.hf
                self.embedding_client = EmbeddingFactory.create(
                    "hf", model_name=emb_config.model_name
                )
            elif self.embedding_type == "openai":
                emb_config = config.embeddings.clients.openai
                self.embedding_client = EmbeddingFactory.create(
                    "openai", 
                    api_key=emb_config.api_key,
                    base_url=emb_config.base_url,
                    model=emb_config.model
                )
            else:
                raise ValueError(f"不支持的embedding类型: {self.embedding_type}")
            
            super()._do_initialize()
        except Exception as e:
            self.logger.error(f"Embedding重排组件 {self.name} 初始化失败: {e}")
            raise
    
    def _rerank(self, query: str, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """使用embedding相似度进行重排"""
        try:
            # 获取查询的embedding
            query_embedding = self.embedding_client.embed_query(query)
            
            # 获取文档的embeddings
            doc_texts = []
            for doc in documents:
                content = doc.get("content", doc.get("text", ""))
                doc_texts.append(content)
            
            doc_embeddings = self.embedding_client.embed_documents(doc_texts)
            
            # 计算相似度
            similarities = self._calculate_similarities(query_embedding, doc_embeddings)
            
            # 创建(文档, 相似度)对
            scored_docs = list(zip(documents, similarities))
            
            return self._filter_by_score(scored_docs)
            
        except Exception as e:
            self.logger.error(f"Embedding重排失败: {e}")
            return documents[:self.top_k]
    
    def _calculate_similarities(self, query_embedding: List[float], doc_embeddings: List[List[float]]) -> List[float]:
        """计算相似度"""
        if self.similarity_metric == "cosine":
            query_vec = np.array(query_embedding).reshape(1, -1)
            doc_vecs = np.array(doc_embeddings)
            similarities = cosine_similarity(query_vec, doc_vecs)[0]
            return similarities.tolist()
        elif self.similarity_metric == "dot_product":
            query_vec = np.array(query_embedding)
            similarities = []
            for doc_vec in doc_embeddings:
                similarity = np.dot(query_vec, np.array(doc_vec))
                similarities.append(similarity)
            return similarities
        elif self.similarity_metric == "euclidean":
            query_vec = np.array(query_embedding)
            similarities = []
            for doc_vec in doc_embeddings:
                # 欧几里得距离转换为相似度（距离越小，相似度越高）
                distance = np.linalg.norm(query_vec - np.array(doc_vec))
                similarity = 1.0 / (1.0 + distance)
                similarities.append(similarity)
            return similarities
        else:
            raise ValueError(f"不支持的相似度计算方法: {self.similarity_metric}")
