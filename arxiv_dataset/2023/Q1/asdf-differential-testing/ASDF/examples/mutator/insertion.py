from crossasr import Mutator
from crossasr.constant import FAILED_TEST_CASE
from crossasr import Text

import random
from random import shuffle
import re
from nltk.corpus import wordnet 


class Insertion(Mutator):

    def __init__(self, name="insertion"):
        super().__init__(name)

    def generate_mutated_sentences(self):
        """
        replace all the error words in the texts to homophone
        """
        mutated_sentences = []
        transcription_results = self.get_transcription_results()
        for i in range(len(transcription_results)): 
            text = transcription_results[i].text
            transcriptions = transcription_results[i].transcriptions
            cases = transcription_results[i].cases
            filename = transcription_results[i].filename

            if FAILED_TEST_CASE in cases.values():
                # set to 10%
                number_of_new_words = max(1, int(0.1*len(text)))
                new_sentence = ' '.join(self.random_insertion(text, number_of_new_words))

                if new_sentence != text:
                    mutated_sentences.append(Text(filename, new_sentence))
        return mutated_sentences

    def random_insertion(self, text: str, n):
        words = text.split(' ')
        words = [word for word in words if word != '']

        for _ in range(n):
            self.add_word(words)
        return words

    def add_word(self, words):
        synonyms = []
        counter = 0
        while len(synonyms) < 1:
            random_word = words[random.randint(0, len(words)-1)]
            synonyms = self.get_synonyms(random_word)
            counter += 1
            if counter >= 10:
                return
        random_synonym = synonyms[0]
        random_idx = random.randint(0, len(words)-1)
        words.insert(random_idx, random_synonym)

    def get_synonyms(self, word):
        synonyms = set()
        for syn in wordnet.synsets(word): 
            for l in syn.lemmas(): 
                synonym = l.name().replace("_", " ").replace("-", " ").lower()
                synonym = "".join([char for char in synonym if char in ' qwertyuiopasdfghjklzxcvbnm'])
                synonyms.add(synonym) 
        if word in synonyms:
            synonyms.remove(word)
        return list(synonyms)
    