from .base_router import BaseRouter
from utils.rag.llm import LLMFactory
from utils.config import config
from typing import List, Any

class LLMRouter(BaseRouter):
    def __init__(self, llm_config: dict = None):
        llm_config = llm_config or config.llm.clients[config.llm.default]
        self.llm = LLMFactory.from_config(llm_config)

    def route(self, query: str, candidates: List[Any], **kwargs) -> Any:
        # 用LLM判断哪个candidate最相关，返回最佳candidate
        prompt = f"请从以下候选项中选择最能回答问题的一个：\n问题：{query}\n候选项：\n"
        for i, c in enumerate(candidates):
            text = getattr(c, 'text', getattr(c, 'get_content', lambda: "")())
            prompt += f"[{i+1}] {text}\n"
        prompt += "\n请直接回复编号，如1、2、3..."
        resp = self.llm.completion(prompt)
        try:
            idx = int(resp.strip().split()[0]) - 1
            return candidates[idx]
        except Exception:
            return candidates[0]  # fallback 