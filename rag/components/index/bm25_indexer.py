import pickle
import os
from typing import Any, Dict, List, Optional
from .base_indexer import BaseIndexer

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    BM25Okapi = None


class BM25IndexerComponent(BaseIndexer):
    """BM25索引器组件"""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        super().__init__(name, config)
        
        if BM25Okapi is None:
            raise ImportError("请安装rank-bm25包: pip install rank-bm25")
            
        # BM25配置
        self.k1 = config.get("k1", 1.2)
        self.b = config.get("b", 0.75)
        self.epsilon = config.get("epsilon", 0.25)
        
        # 存储配置
        self.storage_path = config.get("storage_path", "./data/bm25_index")
        self.auto_save = config.get("auto_save", True)
        
        # 内部状态
        self.bm25 = None
        self.documents = []
        self.doc_metadata = []
        
    def _do_initialize(self):
        """初始化BM25索引器"""
        try:
            # 创建存储目录
            os.makedirs(self.storage_path, exist_ok=True)
            
            # 尝试加载已有索引
            self._load_index()
            
            if self.debug:
                self.logger.debug(f"BM25索引器初始化完成，文档数量: {len(self.documents)}")
                
        except Exception as e:
            self.logger.error(f"初始化BM25索引器失败: {e}")
            raise
            
    def _tokenize(self, text: str) -> List[str]:
        """文本分词"""
        # 简单的分词实现，可以根据需要替换为更复杂的分词器
        import re
        # 移除标点符号并转为小写
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        # 分词
        tokens = text.split()
        return [token for token in tokens if len(token) > 1]
        
    def _get_index_path(self) -> str:
        """获取索引文件路径"""
        return os.path.join(self.storage_path, f"{self.index_name}.pkl")
        
    def _save_index(self):
        """保存索引到文件"""
        try:
            # 构建倒排索引
            index_data = {}
            doc_store = {}
            doc_lengths = {}
            
            for i, doc_meta in enumerate(self.doc_metadata):
                doc_id = doc_meta["id"]
                content = doc_meta["content"]
                tokens = self._tokenize(content)
                
                # 文档存储
                doc_store[doc_id] = {
                    "content": content,
                    "metadata": doc_meta["metadata"]
                }
                
                # 文档长度
                doc_lengths[doc_id] = len(tokens)
                
                # 构建倒排索引
                for token in set(tokens):
                    if token not in index_data:
                        index_data[token] = []
                    if doc_id not in index_data[token]:
                        index_data[token].append(doc_id)
            
            # 计算平均文档长度
            avg_doc_length = sum(doc_lengths.values()) / len(doc_lengths) if doc_lengths else 0
            
            # 保存兼容格式的数据
            save_data = {
                "index_data": index_data,
                "doc_store": doc_store,
                "doc_lengths": doc_lengths,
                "avg_doc_length": avg_doc_length,
                "N": len(self.doc_metadata),
                "config": {
                    "k1": self.k1,
                    "b": self.b,
                    "epsilon": self.epsilon
                }
            }
            
            with open(self._get_index_path(), 'wb') as f:
                pickle.dump(save_data, f)
                
            if self.debug:
                self.logger.debug(f"索引已保存到: {self._get_index_path()}")
                
        except Exception as e:
            self.logger.error(f"保存索引失败: {e}")
            
    def _load_index(self):
        """从文件加载索引"""
        index_path = self._get_index_path()
        if not os.path.exists(index_path):
            if self.debug:
                self.logger.debug("索引文件不存在，将创建新索引")
            return
            
        try:
            with open(index_path, 'rb') as f:
                index_data = pickle.load(f)
                
            # 检查是否是新格式（兼容格式）
            if "index_data" in index_data and "doc_store" in index_data:
                # 新格式：从兼容数据重建BM25对象
                doc_store = index_data["doc_store"]
                self.doc_metadata = []
                all_docs = []
                
                for doc_id, doc_data in doc_store.items():
                    content = doc_data["content"]
                    metadata = doc_data["metadata"]
                    tokens = self._tokenize(content)
                    
                    if tokens:
                        all_docs.append(tokens)
                        self.doc_metadata.append({
                            "id": doc_id,
                            "content": content,
                            "metadata": metadata
                        })
                    
                if all_docs:
                    # 重建BM25对象
                    config_data = index_data.get("config", {})
                    self.bm25 = BM25Okapi(
                        all_docs,
                        k1=config_data.get("k1", self.k1),
                        b=config_data.get("b", self.b),
                        epsilon=config_data.get("epsilon", self.epsilon)
                    )
                    self.documents = all_docs
            else:
                # 旧格式：直接加载
                self.bm25 = index_data.get("bm25")
                self.documents = index_data.get("documents", [])
                self.doc_metadata = index_data.get("doc_metadata", [])
            
            if self.debug:
                self.logger.debug(f"成功加载索引，文档数量: {len(self.documents)}")
            
        except Exception as e:
            self.logger.error(f"加载索引失败: {e}")
            # 重置状态
            self.bm25 = None
            self.documents = []
            self.doc_metadata = []
            
    def create_index(self, index_name: Optional[str] = None, **kwargs) -> bool:
        """创建索引"""
        try:
            # 如果提供了 index_name，更新当前索引名称
            if index_name:
                self.index_name = index_name
                
            # BM25索引在添加文档时自动创建
            if self.debug:
                self.logger.debug(f"BM25索引 {self.index_name} 准备就绪")
            return True
        except Exception as e:
            self.logger.error(f"创建索引失败: {e}")
            return False
            
    # 移除 delete_index 方法
    # def delete_index(self) -> bool:
    #     """删除索引"""
    #     ...
            
    def index_documents(self, documents: List[Dict[str, Any]], index_name: Optional[str] = None) -> bool:
        """索引文档"""
        try:
            # 如果提供了 index_name，更新当前索引名称
            if index_name:
                self.index_name = index_name
                
            # 处理新文档
            new_docs = []
            new_metadata = []
            
            for doc in documents:
                content = doc.get("content", "")
                metadata = doc.get("metadata", {})
                
                # 分词
                tokens = self._tokenize(content)
                if tokens:  # 只添加非空文档
                    new_docs.append(tokens)
                    new_metadata.append({
                        "id": doc.get("id", f"doc_{len(self.documents) + len(new_docs)}"),
                        "content": content,
                        "metadata": metadata
                    })
                    
            if not new_docs:
                if self.debug:
                    self.logger.debug("没有有效文档需要索引")
                return True
                
            # 合并到现有文档
            all_docs = []
            if self.bm25 is not None:
                # 重新构建所有文档的token列表
                for doc_meta in self.doc_metadata:
                    tokens = self._tokenize(doc_meta["content"])
                    if tokens:
                        all_docs.append(tokens)
                        
            all_docs.extend(new_docs)
            self.doc_metadata.extend(new_metadata)
            
            # 重建BM25索引
            self.bm25 = BM25Okapi(
                all_docs, 
                k1=self.k1, 
                b=self.b, 
                epsilon=self.epsilon
            )
            
            # 更新文档列表
            self.documents = all_docs
            
            # 自动保存
            if self.auto_save:
                self._save_index()
                
            if self.debug:
                self.logger.debug(f"成功索引 {len(new_docs)} 个新文档，总文档数: {len(self.documents)}")
                
            return True
            
        except Exception as e:
            self.logger.error(f"索引文档失败: {e}")
            return False
            
    def search(self, query: str, top_k: int = 10, index_name: Optional[str] = None, **kwargs) -> List[Dict[str, Any]]:
        """搜索文档"""
        try:
            # 如果提供了 index_name，更新当前索引名称
            if index_name:
                self.index_name = index_name
                
            if self.bm25 is None or not self.documents:
                if self.debug:
                    self.logger.debug("索引为空，返回空结果")
                return []
                
            # 查询分词
            query_tokens = self._tokenize(query)
            if not query_tokens:
                return []
                
            # BM25搜索
            scores = self.bm25.get_scores(query_tokens)
            
            # 获取top-k结果
            import numpy as np
            top_indices = np.argsort(scores)[::-1][:top_k]
            
            results = []
            for idx in top_indices:
                if idx < len(self.doc_metadata) and scores[idx] > 0:
                    doc_meta = self.doc_metadata[idx]
                    result = {
                        "id": doc_meta["id"],
                        "score": float(scores[idx]),
                        "content": doc_meta["content"],
                        "metadata": doc_meta["metadata"]
                    }
                    results.append(result)
                    
            if self.debug:
                self.logger.debug(f"BM25搜索返回 {len(results)} 个结果")
                
            return results
            
        except Exception as e:
            self.logger.error(f"搜索失败: {e}")
            return []
            
    def get_document(self, doc_id: str, index_name: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """根据ID获取文档"""
        try:
            # 如果提供了 index_name，更新当前索引名称
            if index_name:
                self.index_name = index_name
                
            for doc_meta in self.doc_metadata:
                if doc_meta["id"] == doc_id:
                    return {
                        "id": doc_meta["id"],
                        "content": doc_meta["content"],
                        "metadata": doc_meta["metadata"]
                    }
            return None
            
        except Exception as e:
            self.logger.error(f"获取文档失败: {e}")
            return None