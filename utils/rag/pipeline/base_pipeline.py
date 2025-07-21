from abc import ABC, abstractmethod


class BasePipeline(ABC):
    def __init__(self):
        self._prepared = False

    def prepare(self):
        if self._prepared:
            return
        self._do_prepare()
        self._prepared = True

    def _do_prepare(self):
        """子类实现实际的预处理逻辑"""

    @abstractmethod
    def run(self, query: str):
        pass
