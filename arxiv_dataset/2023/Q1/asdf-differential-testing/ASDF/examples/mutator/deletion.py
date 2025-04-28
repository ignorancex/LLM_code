from crossasr import Mutator
from crossasr.constant import FAILED_TEST_CASE, INDETERMINABLE_TEST_CASE
from crossasr import Text

import random
from random import shuffle
import re
from nltk.corpus import wordnet 

class Deletion(Mutator):

    def __init__(self, name="deletion"):
        super().__init__(name)

    def generate_mutated_sentences(self):
        """
        mutates the sentence by deleting the word before and after the errored word 
        """
        mutated_sentences = []
        failed_transcription_results = self.get_not_mutated_failed_transcription_results()

        for i in range(len(failed_transcription_results)): 
            text_obj = failed_transcription_results[i].get_text()
            text = text_obj.getText()
            filename = text_obj.getFilename()
            transcriptions = failed_transcription_results[i].get_transcriptions()
            cases = failed_transcription_results[i].get_cases()

            error_words = self.get_all_error_words(text, transcriptions, cases)

            for i in range(len(error_words)):
                new_sentence = self.delete_all_error_words(text, [error_words[i]])
                mutated_sentence = ' '.join(new_sentence)

                if mutated_sentence != text:
                    mutated_sentences.append(Text(id=f"{filename}_{str(i+1)}", text=mutated_sentence, original_sentence=text, error_words=[error_words[i]]))

            # if len(error_words) > 0:
            #     new_sentence = self.delete_all_error_words(text, error_words)
            #     mutated_sentence = ' '.join(new_sentence)

            #     if mutated_sentence != text:
            #         mutated_sentences.append(Text(id=filename, text=mutated_sentence, original_sentence=text, error_words=error_words))

        return mutated_sentences

    def delete_all_error_words(self, text, error_words):
        """
        performs deletion before and after all occurances of error words in a sentence
        :returns: new_words []
        """
        words = text.split()
        new_words = [word for word in words if word != '']

        if len(words) < 2:
            return words

        for error_word in error_words:
            new_words = self.delete_from_sentence(new_words, error_word) 
                    
        #if you end up deleting all words, return original sentence
        if len(new_words) == 0:
            return words

        return new_words

    def delete_from_sentence(self, words, error_word):
        """
        deletes the indices before and after every occurance of the error_word
        :returns: words [] """
        indices = [i for i in range(len(words)-1, -1, -1) if words[i]==error_word]

        for index in indices:
            if index == (len(words)-1):
                words.pop(index-1)
            elif index == 0:
                words.pop(index+1)
            else:
                words.pop(index+1)
                words.pop(index-1)
        
        return words

    
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