from abc import ABC, abstractmethod
from typing import Any

class BaseAttack(ABC):
    @abstractmethod
    def __init__(self, *args: Any, **kwargs: Any) -> Any:
        pass
    
    @abstractmethod
    def attack(self, *args: Any, **kwargs: Any) -> Any:
        pass