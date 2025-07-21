from abc import ABC, abstractmethod

class BaseLLM(ABC):
    @abstractmethod
    def completion(self, prompt: str, **kwargs) -> str:
        pass 