from abc import ABC, abstractmethod
from typing import Iterator

class BaseLLM(ABC):

    @abstractmethod
    def completion(self, prompt: str, **kwargs) -> str:
        pass
    
    @abstractmethod
    def completion_stream(self, prompt: str, **kwargs) -> Iterator[str]:
        """流式生成接口"""
        pass