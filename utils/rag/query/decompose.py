from .base_query_expansion import BaseQueryExpansion
from utils.llm import LLMFactory
from utils.prompts.query_expansion import DECOMPOSE_PROMPT
import re
from typing import Union, List

class DecomposeExpansion(BaseQueryExpansion):
    def __init__(self):
        self.llm = LLMFactory.from_config()

    def transform(self, query: str) -> List[str]:
        prompt = DECOMPOSE_PROMPT.format(query=query)
        resp = self.llm.completion(prompt)
        match = re.search(r'<output>(.*?)</output>', resp, re.DOTALL)
        content = match.group(1).strip() if match else resp.strip()
        return [line.strip() for line in content.splitlines() if line.strip()] 