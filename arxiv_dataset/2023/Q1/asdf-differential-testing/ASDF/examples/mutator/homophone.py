from crossasr.utils import preprocess_text
from crossasr import Mutator
from crossasr.constant import FAILED_TEST_CASE
from crossasr import Word
from jiwer import wer
from wordhoard import Homophones
from crossasr import Text
import requests
import nltk
from functools import lru_cache
from itertools import product as iterprod
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import eng_to_ipa as ipa
from dotenv import load_dotenv
import os
import spacy

load_dotenv("./.env")
OXFORD_APP_ID  = os.getenv("OXFORD_APP_ID")
OXFORD_APP_KEY  = os.getenv("OXFORD_APP_KEY")

class Homophone(Mutator):

    def __init__(self, name="homophone"):
        super().__init__(name)
        self.homophone_history = set()  # To prevent duplicated homophone sentence generation in different iterations


    def generate_mutated_sentences(self):
        """
        get sentences from the homophones of the error words
        """
        mutated_sentences = []
        all_homophones = []
        
        transcription_results = self.get_not_mutated_failed_transcription_results()
        for i in range(len(transcription_results)): 
            text_obj = transcription_results[i].get_text()

            text = text_obj.getText()
            filename = text_obj.getFilename()
            transcriptions = transcription_results[i].get_transcriptions()
            cases = transcription_results[i].get_cases()

            error_words = self.get_all_error_words(text, transcriptions, cases)
            self.homophone_history.update(error_words)

            for word in error_words:
                homophones = self.get_homophones(word)
                for homophone in homophones:
                    if homophone not in self.homophone_history:
                        all_homophones.append({'homophone': homophone, 'filename': filename, 'original_sentence': text, 'error_word': word})
                        self.homophone_history.add(homophone)

        for homophone_dict in all_homophones:
            homophone, filename, original_sentence, error_word = homophone_dict['homophone'], homophone_dict['filename'],homophone_dict['original_sentence'], homophone_dict['error_word']
            new_sentence = self.get_sentence_from_free_dictionary(homophone)
            if new_sentence != None:
                processed_new_sentence = preprocess_text(new_sentence)
                if processed_new_sentence != "":
                    mutated_sentences.append(Text(id=f"{filename}_{homophone}", text=processed_new_sentence, original_sentence=original_sentence,error_words=[error_word]))
                    print(processed_new_sentence)
            
        return mutated_sentences

    def get_sentence_from_oxford_dictionary(self, homophone):
        app_id  = OXFORD_APP_ID
        app_key  = OXFORD_APP_KEY
        endpoint = "entries"
        language_code = "en-us"
        word_id = homophone
        url = "https://od-api.oxforddictionaries.com/api/v2/" + endpoint + "/" + language_code + "/" + word_id.lower()
        r = requests.get(url, headers = {"app_id": app_id, "app_key": app_key})
        
        # Some words might not have example sentence thus will cause KeyError
        try:
            sentence = r.json()['results'][0]['lexicalEntries'][0]['entries'][0]['senses'][0]['examples'][0]['text']
        except:
            sentence = None
        return sentence

    def get_sentence_from_free_dictionary(self, homophone):
        url = "https://api.dictionaryapi.dev/api/v2/entries/en/"
        r = requests.get(url + homophone)
        try:
            obj = r.json()[0]

            examples = []

            for meaning in obj["meanings"]:
                for definition in meaning["definitions"]:
                    if definition.get("example") is not None:
                        examples.append(definition.get("example"))

            if len(examples) != 0:
                return examples[0]
            # Case where definition found but no example sentences
            else:
                return None
        
        # Case where no definition found
        except:
            return None

    def get_homophones(self, word: str):
        homophone = Homophones(search_string=word)
        homophone_result = homophone.find_homophones()
        if isinstance(homophone_result, list):
            output = list(map(lambda x: x.split()[-1], homophone_result))
        else:
            output = []
        return output

    def get_rhymes(self, word: str):
        rhymes = ipa.get_rhymes(word)
        return rhymes
