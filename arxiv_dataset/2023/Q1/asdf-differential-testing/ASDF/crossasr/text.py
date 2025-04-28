from distutils.log import error
import functools
from typing import List

@functools.total_ordering
class Text:
    def __init__(self, id: int, text: str, is_mutated: bool = False, original_sentence: str = None, error_words: List[str] = None):
        self.id = id
        self.text = text
        self.is_mutated = is_mutated
        self.original_sentence = original_sentence
        self.error_words = error_words

    def __eq__(self, other):
        return self.id == other.id and self.text == other.text

    def __lt__(self, other):
        return (self.id, self.text) < (other.id, other.text)

    def getId(self):
        return self.id

    def getFilename(self):
        return str(self.getId())

    def setId(self, id: int):
        self.id = id

    def getText(self):
        return self.text

    def setText(self, text: str):
        self.text = text

    def getIsMutated(self):
        return self.is_mutated

    def setIsMutated(self, is_mutated: bool):
        self.is_mutated = is_mutated

    def getOriginalSentence(self):
        return self.original_sentence

    def setOriginalSentence(self, original_sentence: str):
        self.original_sentence = original_sentence

    def getErrorWords(self):
        return self.error_words.copy()

    def setErrorWords(self, error_words: str):
        self.error_words = error_words.copy()
