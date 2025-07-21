from abc import ABC, abstractmethod
from typing import Any, List

class BaseRouter(ABC):
    @abstractmethod
    def route(self, query: str, candidates: List[Any], **kwargs) -> Any:
        pass 