from typing import List

class Word:
  def __init__(self, word, upos):
        pos_dict = {
          "NOUN":"noun",
          "PROPN":"noun",
          "VERB":"verb",
          "ADJ":"adjective",
          "ADV":"adverb",
          "ADP":"preposition",
          "DET":"determiner",
          "PRON":"pronoun",
          "CCONJ":"conjunction",
          "SCONJ":"conjunction",
          "INTJ":"interjection"
        }

        self.word = word
        self.upos = upos
        self.phonemes = []
        self.phonetic = ""

        # If spaCy identifies the word as something other than the 9 Parts of Speech, set POS as "OTHER"
        self.pos = pos_dict.get(self.upos,"OTHER")
  
  def get_word(self):
        return self.word

  def set_word(self, word: str):
        self.word = word

  def get_pos(self):
        return self.pos

  def set_pos(self, pos: str):
        self.pos = pos

  def get_phonetic(self):
        return self.phonetic

  def set_phonetic(self, phonetic: str):
        self.phonetic = phonetic

  def get_phonemes(self):
        return self.phonemes

  def set_phonemes(self, phonemes: List[str]):
        self.phonemes = phonemes