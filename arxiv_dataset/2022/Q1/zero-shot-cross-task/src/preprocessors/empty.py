from copy import deepcopy
from typing import Dict
import math

from src.preprocessors.task_preprocessor import FixedChoiceTaskPreprocessor


@FixedChoiceTaskPreprocessor.register("empty")
class EmptyPreprocessor(FixedChoiceTaskPreprocessor):
    def __init__(
            self
    ):
        super().__init__(
            choices=["Choice"],
            classification_template="{}",
            premise_template="{}",
            hypothesis_template="{}",
            question_template="{}",
            context_template="{}",
            choice_str="",
            mcq_choice_str="",
            is_mcq=False,
            dont_add_extra_text=False
        )

    def _process_instance(self, data_instance, idx) -> Dict:
        return {
            "choices"       : self.choices,
            "idx"           : idx,
            "domain"        : "",
            "input_sequence": "",
            "label"         : 0
        }

    def convert_to_classification(self, processed_instance: Dict) -> Dict:
        return processed_instance

    def convert_to_entailment(self, processed_instance: Dict) -> Dict:
        processed_instance['premise'] = processed_instance.pop("input_sequence")
        processed_instance['hypothesis'] = processed_instance['premise']
        return processed_instance

    def convert_to_qa(self, processed_instance: Dict) -> Dict:
        processed_instance['context'] = processed_instance.pop("input_sequence")
        processed_instance['question'] = processed_instance['context']
        return processed_instance
