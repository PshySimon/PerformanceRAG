from .base import BaseLLM
import requests
import os

class OpenAILLM(BaseLLM):
    def __init__(self, model, api_key, base_url, temperature=0.9, max_tokens=4096, **kwargs):
        self.model = model
        self.api_key = os.environ.get("OPENAI_API_KEY", api_key)
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.max_tokens = max_tokens

    def completion(self, prompt: str, **kwargs) -> str:
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }
        data.update(kwargs)
        resp = requests.post(url, headers=headers, json=data, timeout=60)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"] 