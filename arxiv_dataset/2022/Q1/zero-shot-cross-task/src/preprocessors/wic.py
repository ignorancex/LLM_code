from copy import deepcopy
from typing import Dict
import math

from src.preprocessors.task_preprocessor import FixedChoiceTaskPreprocessor


@FixedChoiceTaskPreprocessor.register("wic")
class WICPreprocessor(FixedChoiceTaskPreprocessor):
    def __init__(
            self,
            choices=None,
            classification_template: str = None,
            premise_template: str = None,
            hypothesis_template: str = None,
            question_template: str = None,
            context_template: str = None,
            is_mcq: bool = False,
            choice_str: str = None,
            mcq_choice_str: str = None,
            dont_add_extra_text: bool = False
    ):
        choices = choices or ["No", "Yes"]
        assert len(choices) == 2

        classification_template = (
                classification_template
                or "Sentence 1: {}. Sentence 2: {}. The word \"{}\" has the "
                   "same meaning in both."
        )
        premise_template = (
                premise_template
                or "Sentence 1: {}. Sentence 2: {}."
        )
        hypothesis_template = (
                hypothesis_template
                or "The word \"{}\" has the same meaning in both."
        )
        context_template = (
                context_template
                or "Sentence 1: {}. Sentence 2: {}."
        )
        question_template = (
                question_template
                or "Does the word \"{}\" have the same meaning in both?"
        )

        super().__init__(
            choices=choices,
            classification_template=classification_template,
            premise_template=premise_template,
            hypothesis_template=hypothesis_template,
            question_template=question_template,
            context_template=context_template,
            choice_str=choice_str,
            mcq_choice_str=mcq_choice_str,
            is_mcq=is_mcq,
            dont_add_extra_text=dont_add_extra_text
        )
        if self.dont_add_extra_text:
            self.classification_template = "{} {} {}"
            self.premise_template = "{} {}"
            self.hypothesis_template = "{}"
            self.context_template = "{} {}"
            self.question_template = "{}"

    def _process_instance(self, data_instance, idx) -> Dict:
        return {
            "choices"   : self.choices,
            "idx"       : idx,
            "domain"    : "word sense",
            "sentence_1": data_instance['sentence1'],
            "sentence_2": data_instance['sentence2'],
            "word"      : data_instance['word'],
            "label"     : data_instance['label']
        }

    def convert_to_classification(self, processed_instance: Dict) -> Dict:
        processed_instance['input_sequence'] = self.classification_template.format(
            processed_instance.pop('sentence_1'), processed_instance.pop('sentence_2'), processed_instance.pop('word')
        )
        return processed_instance

    def convert_to_entailment(self, processed_instance: Dict) -> Dict:
        processed_instance['premise'] = self.premise_template.format(
            processed_instance.pop('sentence_1'), processed_instance.pop('sentence_2')
        )
        processed_instance['hypothesis'] = self.hypothesis_template.format(
            processed_instance.pop('word')
        )
        return processed_instance

    def convert_to_qa(self, processed_instance: Dict) -> Dict:
        processed_instance['context'] = self.context_template.format(
            processed_instance.pop('sentence_1'), processed_instance.pop('sentence_2')
        )
        processed_instance['question'] = self.question_template.format(
            processed_instance.pop('word')
        )
        return processed_instance
