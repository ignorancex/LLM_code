from abc import ABC, abstractmethod
from typing import Sequence


class AbstractBaseLogger(ABC):
    @abstractmethod
    def log(self, log_data, step, commit=False):
        raise NotImplementedError

    @abstractmethod
    def complete(self, log_data, step):
        raise NotImplementedError

class LoggingService():
    def __init__(self, loggers: Sequence[AbstractBaseLogger]):
        self.loggers = loggers

    def log(self, log_data, step, commit=False):
        for logger in self.loggers:
            logger.log(log_data, step, commit)

    def complete(self, log_data, step):
        for logger in self.loggers:
            logger.complete(log_data, step)

