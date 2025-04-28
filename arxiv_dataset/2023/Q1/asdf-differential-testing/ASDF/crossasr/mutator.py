import json
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import ceil
from typing import Dict, List
from copy import deepcopy
from crossasr.constant import FAILED_TEST_CASE
from crossasr.transcription_result import TranscriptionResult
from crossasr.text import Text
from crossasr.word import Word
import spacy

class Mutator:
    def __init__(self, name):
        self.name = name
        self.iteration = 1
        self.transcription_results_dict: Dict[int, List[TranscriptionResult]] = {self.iteration: []}
        self.phoneme_dict = {}
        self.nlp = spacy.load("en_core_web_sm")

    def get_phoneme_dict(self):
        return self.phoneme_dict

    def set_phoneme_dict(self, dict):
        self.phoneme_dict = dict

    def get_name(self):
        return self.name

    def set_name(self, name: str):
        self.name = name

    def count_frequency(self, my_list, count_dict):
        """
        Counts frequency of words in a list and creates a Frequency Dictionary
        """
        for i in my_list:
            count_dict[i] = count_dict.get(i, 0) + 1
        return count_dict
    
    def get_error_phonemes(self, sentence, error_words):
        wordlist = []
        phonemesList = ['ɓ', 'ʙ', 'β', 'ɕ', 'ç', 'ɗ', 'ɖ', 'ð', 'ʤ', 'ɟ', 'ʄ', 'ɡ', 'ɠ', 'ɢ', 'ʛ', 'ɦ', 'ɧ', 'ħ', 'ʜ', 'ʝ', 'ɭ', 'ɬ', 'ɫ', 'ɮ', 'ʟ', 'ʎ', 'ɱ', 'ɰ', 'ŋ', 'ɳ', 'ɲ', 'ɴ', 'ɸ', 'ɹ', 'ɻ', 'ɺ', 'ɾ', 'ɽ', 'ʀ', 'ʁ', 'ʂ', 'ʃ', 'θ', 'ʈ', 'ʧ', 'ʋ', 'ɣ', 'ʍ', 'χ', 'ʑ', 'ʐ', 'ʒ', 'b', 'd', 'f', 'g', 'h', 'dʒ', 'k', 'l', 'm', 'n', 'p', 'r', 's', 't', 'v', 'w', 'z', 'tʃ', 'j', 'ɑ', 'ɐ', 'ɒ', 'æ', 'ə', 'ɘ', 'ɵ', 'ɚ', 'ɛ', 'ɜ', 'ɝ', 'ɞ', 'ɨ', 'ɪ', 'ɔ', 'ø', 'œ', 'ɶ', 'ɥ', 'ʌ', 'ʊ', 'ʉ', 'ɯ', 'ɤ', 'ʏ', 'eɪ', 'e', 'iː', 'aɪ', 'oʊ', 'uː', 'ɔɪ', 'aʊ', 'eəʳ', 'ɑː', 'ɜːʳ', 'ɔː', 'ɪəʳ', 'ʊəʳ', 'ʔ', 'ʡ', 'ʕ', 'ʢ', 'ʘ', 'ǀ', 'ǃ', 'ǂ', 'ǁ']
        doc = self.nlp(sentence)
        for token in doc:
            if token.text in error_words:
                word = Word(token.text, token.pos_) 
                wordlist.append(word)
        
        for word in wordlist:
            keyword = word.get_word()

            url = "https://api.dictionaryapi.dev/api/v2/entries/en/"

            try:
                r = requests.get(url + keyword)
                for i in r.json():
                    for j in i["meanings"]:
                        # If spaCy cannot identify the POS of the word, take any phonetic of the word
                        if j["partOfSpeech"] == word.get_pos() or word.get_pos() == "OTHER":
                            word.set_phonetic(i["phonetic"].replace("/","").replace("ˈ",""))

                print("Phonetic of " + word.get_word() + " in " + word.get_pos() + " form is " + word.get_phonetic())
                curr_phoneme = [*word.get_phonetic()]
                
                for i in curr_phoneme:
                    if i not in phonemesList:
                        curr_phoneme.remove(i)
                
                word.set_phonemes(curr_phoneme)
                print(word.get_phonemes())

            except:
                print(word.get_word() + " has no phonetic")
        
        error_phonemes = [word.get_phonemes() for word in wordlist]
        error_phonemes = [item for sublist in error_phonemes for item in sublist]
        return error_phonemes

    

    def generate_phoneme_graph(self, result_filepath):
        def add_value_labels(ax, spacing=5):
            # For each bar: Place a label
            for rect in ax.patches:
                # Get X and Y placement of label from rect.
                y_value = rect.get_height()
                x_value = rect.get_x() + rect.get_width() / 2

                # Number of points between bar and label. Change to your liking.
                space = spacing

                # Create annotation
                ax.annotate(
                    y_value,                      # Use `label` as label
                    (x_value, y_value),         # Place label at end of the bar
                    xytext=(0, space),          # Vertically shift label by `space`
                    textcoords="offset points", # Interpret `xytext` as offset in points
                    ha='center',                # Horizontally center label
                    va='bottom')                

        for transcription_res_list in self.transcription_results_dict.values():
            for transcription_res in transcription_res_list:
                text_obj = transcription_res.get_text()

                text = text_obj.getText()
                transcriptions = transcription_res.get_transcriptions()
                cases = transcription_res.get_cases()

                error_words = self.get_all_error_words(text, transcriptions, cases)

                wordlist = self.get_error_phonemes(text, error_words)
                self.phoneme_dict = self.count_frequency(wordlist, self.phoneme_dict)
        
        df = pd.DataFrame([self.phoneme_dict])
        df = df.T
        df.columns = ["Frequency"]
        print(df)

        min_y = min(self.phoneme_dict.values())
        max_y = max(self.phoneme_dict.values())
        print("MIN_Y: " + str(min_y))
        print("MAX_Y: " + str(max_y))
        if min_y + max_y != 0:
            y_tick = ceil(max_y/10)

            yticks = np.arange(0, max_y+1, y_tick)

            ax =  df.plot(kind = "bar", figsize = (30,10), yticks = yticks)
            add_value_labels(ax)

            plt.savefig(result_filepath)

        else:
            print("No plot due to no phonemes found.")

        

    def reset(self):
        self.iteration += 1
        self.transcription_results_dict[self.iteration] = []

    def save_transcription_result(self, transcription_result: TranscriptionResult):
        self.transcription_results_dict[self.iteration].append(transcription_result)

    def get_transcription_results(self) -> List[TranscriptionResult]:
        return deepcopy(self.transcription_results_dict[self.iteration])

    def get_all_failed_transcription_results(self) -> List[TranscriptionResult]:
        failed_transcription_results = []
        for result in self.get_transcription_results():
            if FAILED_TEST_CASE in result.get_cases().values():
                failed_transcription_results.append(result)
        return failed_transcription_results

    def get_not_mutated_failed_transcription_results(self) -> List[TranscriptionResult]:
        failed_transcription_results = []
        for result in self.get_transcription_results():
            is_mutated = result.get_text().getIsMutated()
            if not is_mutated and FAILED_TEST_CASE in result.cases.values():
                failed_transcription_results.append(result)
        return failed_transcription_results

    def get_failed_test_cases_analysis(self, number_of_asr: int):

        total_corpus_text = 0
        total_transformed_text = 0
        total_corpus_failed_text = 0
        total_transformed_failed_text = 0
        total_corpus_failed_cases = 0
        total_transformed_failed_cases = 0

        for i in range(1, self.iteration + 1):
            transcription_results = self.transcription_results_dict[i]
            for result in transcription_results:
                cases_values = result.get_cases().values()
                is_mutated = result.get_text().getIsMutated()
                if not is_mutated:
                    total_corpus_text += 1
                    if FAILED_TEST_CASE in cases_values:
                        total_corpus_failed_text += 1
                        total_corpus_failed_cases += list(cases_values).count(FAILED_TEST_CASE)
                else:
                    total_transformed_text += 1
                    if FAILED_TEST_CASE in cases_values:
                        total_transformed_failed_text += 1
                        total_transformed_failed_cases += list(cases_values).count(FAILED_TEST_CASE)
        
        
        failed_test_cases_analysis = {
            "total_corpus_text": total_corpus_text,
            "total_corpus_cases": total_corpus_text * number_of_asr if total_corpus_text > 0 else 0,
            "total_corpus_failed_text": total_corpus_failed_text,
            "total_corpus_failed_cases": total_corpus_failed_cases,
            "total_transformed_text": total_transformed_text,
            "total_transformed_failed_text": total_transformed_failed_text,
            "total_transformed_failed_cases": total_transformed_failed_cases,
            "corpus_failed_text_percentage": round(total_corpus_failed_text / total_corpus_text, 3) if total_corpus_text > 0 else 0,  # to prevent ZeroDivisionError
            "transformed_failed_text_percentage": round(total_transformed_failed_text / total_transformed_text, 3) if total_transformed_text > 0 else 0,
            "corpus_failed_cases_percentage": round(total_corpus_failed_cases / (total_corpus_text * number_of_asr), 3) if total_corpus_text > 0 else 0,
            "transformed_failed_cases_percentage": round(total_transformed_failed_cases / (total_transformed_text * number_of_asr), 3) if total_transformed_text > 0 else 0,
        }

        return failed_test_cases_analysis

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

    def wrap_mutated_sentences(self, mutated_sentences: List[Text]):
        for sentence in mutated_sentences:
            sentence.setId(f'm_{sentence.getId()}')
            sentence.setIsMutated(True)

    def generate_wrapped_mutated_sentences(self):
        mutated_texts = self.generate_mutated_sentences()
        self.wrap_mutated_sentences(mutated_texts)

        return mutated_texts

    def generate_mutated_sentences(self):
        """
        Generate mutated sentences
        This is an abstract function that needs to be implemented by the child class
        """
        raise NotImplementedError()

    def save_all_test_cases(self, filepath: str):
        with open(filepath, 'w') as outfile:
            json.dump(self.transcription_results_dict, outfile, default=lambda o: o.__dict__, 
                sort_keys=True, indent=2)
