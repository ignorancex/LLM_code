from abc import ABC, abstractmethod
from typing import Any

class BasePreprocessor(ABC):
    @abstractmethod
    def __init__(self, config):
        pass
    
    @abstractmethod
    def preprocess(self, *args: Any, **kwargs: Any) -> Any:
        pass