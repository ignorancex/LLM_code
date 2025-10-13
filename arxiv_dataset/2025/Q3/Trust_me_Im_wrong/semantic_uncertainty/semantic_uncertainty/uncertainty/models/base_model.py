from abc import ABC, abstractmethod
from typing import List, Text


STOP_SEQUENCES = ['\n\n\n\n', '\n\n\n', "\n\n", 'Question:', 'Context:', ".\n", ". ",  'question:', "Alice", "Bob", "(", "Explanation", "\nquestion:", "What", "\nanswer"]


class BaseModel(ABC):

    stop_sequences: List[Text]

    @abstractmethod
    def predict(self, input_data, temperature):
        pass

    @abstractmethod
    def get_p_true(self, input_data):
        pass
