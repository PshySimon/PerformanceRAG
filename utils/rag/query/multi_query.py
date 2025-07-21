from .base_query_expansion import BaseQueryExpansion
from utils.llm.factory import LLMFactory
from utils.prompts.query_expansion import MULTI_QUERY_PROMPT
import re
from typing import List


class MultiQueryExpansion(BaseQueryExpansion):
    def __init__(self, n: int = 3):
        self.llm = LLMFactory.from_config()
        self.n = n

    def transform(self, query: str) -> List[str]:
        prompt = MULTI_QUERY_PROMPT.format(query=query, n=self.n)
        resp = self.llm.completion(prompt)
        match = re.search(r"<output>(.*?)</output>", resp, re.DOTALL)
        content = match.group(1).strip() if match else resp.strip()
        return [line.strip() for line in content.splitlines() if line.strip()]
