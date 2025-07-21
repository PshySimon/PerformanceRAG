from abc import ABC, abstractmethod
from typing import Union, List

class BaseQueryExpansion(ABC):
    @abstractmethod
    def transform(self, query: str) -> Union[str, List[str]]:
        pass 