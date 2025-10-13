from abc import ABC, abstractmethod

class BaseModel(ABC):
    @abstractmethod
    def __init__(self, config):
        pass
    