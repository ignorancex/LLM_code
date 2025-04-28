from crossasr import Mutator
from crossasr.constant import FAILED_TEST_CASE, INDETERMINABLE_TEST_CASE
from crossasr import Text

import random
from random import shuffle
import re
from nltk.corpus import wordnet 


class Swap(Mutator):

    def __init__(self, name="swap"):
        super().__init__(name)

    def generate_mutated_sentences(self):
        """
        mutates the sentence by swapping the word before and after the errored words
        """
        mutated_sentences = []
        transcription_results = self.get_transcription_results()
        for i in range(len(transcription_results)): 
            text = transcription_results[i].text
            transcriptions = transcription_results[i].transcriptions
            cases = transcription_results[i].cases
            filename = transcription_results[i].filename

            if INDETERMINABLE_TEST_CASE not in cases.values():
                error_words = self.get_all_error_words(text, transcriptions, cases)

                if len(error_words) > 0: 
                    new_sentence = self.swap_words(text, error_words)
                    mutated_sentence = ' '.join(new_sentence)

                    if mutated_sentence != text:
                        mutated_sentences.append(Text(filename, mutated_sentence))

        return mutated_sentences

    def swap_words(self, text, error_words):
        words = text.split(' ')
        new_words = [word for word in words if word != '']

        if len(words) < 2:
            return words

        for word in error_words:
            indices = [i for i in range(len(new_words)) if new_words[i]==word]

            for index in indices:
                if (index != (len(new_words)-1)) and (index != 0):
                    new_words[index-1], new_words[index+1] = new_words[index+1], new_words[index-1]

        return new_words
    
    def get_all_error_words(self, text, transcriptions, cases):
        """
        get all error texts from transcriptions
        :returns: all_error_words []
        """

        def get_error_words(text, transcription):
            text_words = text.split()
            transcription_words = transcription.split()

            word_count = {}

            # Using dictionary to determine which word in the transcription doesn't exist in text
            for word in text_words:
                if word not in word_count:
                    word_count[word] = 1
                else:
                    word_count[word] += 1

            for word in transcription_words:
                if word in word_count:
                    word_count[word] -= 1
            
            error_words = []
            for k in word_count.keys():
                if word_count[k] > 0:
                    error_words.append(k)
            
            return error_words

        all_error_words = []

        for k, transcription in transcriptions.items():
            case = cases[k]
            if case == 1 and transcription != "":
                error_words = get_error_words(text, transcription)
                all_error_words += error_words
        
        all_error_words = list(set(all_error_words))  # remove duplicates

        return all_error_words
    