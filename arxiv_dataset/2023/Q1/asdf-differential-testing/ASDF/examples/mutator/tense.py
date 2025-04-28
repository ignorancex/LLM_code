import spacy
import pyinflect# importing sys
import sys

from crossasr import Mutator
from crossasr import Text
from crossasr.constant import FAILED_TEST_CASE, INDETERMINABLE_TEST_CASE

class Tense(Mutator):

    def __init__(self, name="tense"):
        super().__init__(name)
        # init spaCy pipeline
        self.model = spacy.load("en_core_web_sm")

    def generate_mutated_sentences(self):
        """
        replace all verbs in text with past tense
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
            mutated_sentence = self.transform_tense(text)

            # no plurals and new sentence is different
            if (mutated_sentence is not None) and (mutated_sentence != text):
                mutated_sentences.append(Text(id=filename, text=mutated_sentence, is_mutated=True, original_sentence=text, error_words=error_words))
        return mutated_sentences

    def transform_tense(self, text: str):
        doc_dep = self.model(text)
        VERBS = ["VB", "VBD", "VBG", "VBN", "VBP", "VBZ"]

        for i in range(len(doc_dep)):
            token = doc_dep[i]
            # replace all types of verbs, disregarding grammar
            if (token.tag_ in VERBS) and (token._.inflect("VBD")):
                text = text.replace(' '+token.text+' ', ' '+token._.inflect("VBD")+' ')
        return text
        