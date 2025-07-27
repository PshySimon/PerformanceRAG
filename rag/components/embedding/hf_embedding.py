from typing import List, Optional, Union

import torch
from transformers import AutoModel, AutoTokenizer

from .base_embedding import BaseEmbedding


class HFEmbedding(BaseEmbedding):
    """HuggingFace embedding客户端，支持单条和批量文本embedding"""

    def __init__(
        self,
        model_name: str,
        device: Optional[str] = None,
        batch_size: int = 32,
        max_length: int = 512,
        normalize_embeddings: bool = True,
        dimensions: int = None,
    ):
        """初始化HuggingFace embedding客户端

        Args:
            model_name: 模型名称或路径
            device: 设备，可以是'cpu', 'cuda', 'mps'等，如果为None则自动选择
            batch_size: 批处理大小
            max_length: 最大序列长度
            normalize_embeddings: 是否对嵌入向量进行归一化
            dimensions: 嵌入向量维度，如果为None则从模型配置中获取
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.normalize_embeddings = normalize_embeddings

        # 加载分词器和模型
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

        # 获取embedding维度
        if dimensions is None:
            dimensions = self.model.config.hidden_size

        super().__init__(dimensions=dimensions)

        # 设置设备
        if device:
            self.device = device
        elif torch.cuda.is_available():
            self.device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        self.model.to(self.device)
        self.model.eval()

    def _normalize(
        self, embeddings: Union[torch.Tensor, List[float]]
    ) -> Union[torch.Tensor, List[float]]:
        """对嵌入向量进行归一化

        Args:
            embeddings: 嵌入向量

        Returns:
            归一化后的嵌入向量
        """
        if not self.normalize_embeddings:
            return embeddings

        if isinstance(embeddings, torch.Tensor):
            return embeddings / embeddings.norm(dim=-1, keepdim=True)
        else:
            import numpy as np

            embeddings_np = np.array(embeddings)
            norm = np.linalg.norm(embeddings_np)
            if norm > 0:
                embeddings_np = embeddings_np / norm
            return embeddings_np.tolist()

    @torch.no_grad()
    def embed_text(self, text: str) -> List[float]:
        """对单条文本进行embedding

        Args:
            text: 需要嵌入的文本

        Returns:
            List[float]: 嵌入向量
        """
        if not text.strip():
            return [0.0]  # 返回空向量

        # 对文本进行编码
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=self.max_length
        ).to(self.device)

        # 获取模型输出
        outputs = self.model(**inputs)

        # 使用[CLS]令牌的最后一层隐藏状态作为嵌入向量
        embedding = outputs.last_hidden_state[:, 0, :].squeeze(0).cpu().numpy()

        # 归一化
        embedding = self._normalize(embedding)

        return embedding.tolist()

    @torch.no_grad()
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """对多条文本进行批量embedding

        Args:
            texts: 需要嵌入的文本列表

        Returns:
            List[List[float]]: 嵌入向量列表
        """
        if not texts:
            return []  # 返回空列表

        # 过滤空文本
        non_empty_indices = [i for i, text in enumerate(texts) if text.strip()]
        if not non_empty_indices:
            return [[0.0]] * len(texts)  # 全部是空文本，返回空向量列表

        non_empty_texts = [texts[i] for i in non_empty_indices]
        results = []

        # 批量处理
        for i in range(0, len(non_empty_texts), self.batch_size):
            batch = non_empty_texts[i : i + self.batch_size]

            # 对文本进行编码
            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            ).to(self.device)

            # 获取模型输出
            outputs = self.model(**inputs)

            # 使用[CLS]令牌的最后一层隐藏状态作为嵌入向量
            embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()

            # 归一化
            embeddings = [self._normalize(emb) for emb in embeddings]

            results.extend([emb.tolist() for emb in embeddings])

        # 处理原始文本中的空文本
        final_results = []
        result_idx = 0
        for i in range(len(texts)):
            if i in non_empty_indices:
                final_results.append(results[result_idx])
                result_idx += 1
            else:
                final_results.append([0.0])  # 空文本返回空向量

        return final_results
