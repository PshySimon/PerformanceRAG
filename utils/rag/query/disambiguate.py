from .base_query_expansion import BaseQueryExpansion
from utils.llm import LLMFactory
from utils.prompts.query_expansion import DISAMBIGUATE_PROMPT
import re
from typing import Union, List

class DisambiguateExpansion(BaseQueryExpansion):
    def __init__(self):
        self.llm = LLMFactory.from_config()

    def transform(self, query: str) -> str:
        prompt = DISAMBIGUATE_PROMPT.format(query=query)
        resp = self.llm.completion(prompt)
        match = re.search(r'<output>(.*?)</output>', resp, re.DOTALL)
        return match.group(1).strip() if match else resp.strip() 