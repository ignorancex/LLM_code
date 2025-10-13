from abc import ABC, abstractmethod

class BaseLoader(ABC):
    @abstractmethod
    def __init__(self, config):
        pass

    @abstractmethod
    def __len__(self):
        pass
    
    @abstractmethod
    def __getitem__(self, idx):
        pass