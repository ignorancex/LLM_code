from abc import ABC, abstractmethod
from typing import Any

class BaseJudge(ABC):
    @abstractmethod
    def __init__(self, *args: Any, **kwargs: Any) -> Any:
        pass

    @abstractmethod
    def score(self, *args: Any, **kwargs: Any) -> Any:
        pass

    @abstractmethod
    def judge(self, *args: Any, **kwargs: Any) -> Any:
        pass