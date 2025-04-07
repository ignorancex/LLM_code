from copy import deepcopy
from typing import Dict
import math

from src.preprocessors.task_preprocessor import FixedChoiceTaskPreprocessor


@FixedChoiceTaskPreprocessor.register('aqua_rat')
class AquaRatPreprocessor(FixedChoiceTaskPreprocessor):

    def __init__(
            self,
            use_lowercase_choices: bool = False,
            is_mcq: bool = False,
            choice_str: str = None,
            mcq_choice_str: str = None,
            keep_choices_in_answers: bool = False,
            dont_add_extra_text: bool = False,
    ):
        self.use_lowercase_choices = use_lowercase_choices
        self.keep_choices_in_answers = keep_choices_in_answers
        if self.use_lowercase_choices:
            choices = ["a", "b", "c", "d", "e"]
        else:
            choices = ["A", "B", "C", "D", "E"]

        classification_template = "{} Possible answers: {}"
        premise_template = "{}"
        hypothesis_template = "Choices are: {}"
        context_template = "Choices are: {}"
        question_template = "{}"

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

        self._correct_to_int = {k: i for i, k in enumerate(["A", "B", "C", "D", "E"])}

        if self.dont_add_extra_text:
            self.classification_template = "{} {}"
            self.premise_template = "{}"
            self.hypothesis_template = "{}"
            self.context_template = "{}"
            self.question_template = "{}"

    def _process_instance(self, data_instance, idx) -> Dict:
        output_dict = {
            "choices" : self.choices,
            "idx"     : idx,
            "domain"  : "math problem",
            "question": data_instance['question'],
            "label"   : self._correct_to_int[data_instance['correct']]
        }

        choice_strs = []
        mcq_choices = []

        for choice in data_instance['options']:
            choice_letter, answer = choice.split(")", 1)

            if self.use_lowercase_choices:
                mcq_choice_str = f"{choice_letter.lower()}) {answer}"
            else:
                mcq_choice_str = choice.replace(")", ") ", 1)
            choice_str = answer.strip()
            choice_strs.append(choice_str)
            mcq_choices.append(mcq_choice_str)

        output_dict['possible_answers'] = ", ".join(choice_strs)
        if self.is_mcq:
            output_dict['choice_string'] = '\n'.join(mcq_choices)
        elif self.keep_choices_in_answers:
            output_dict['possible_answers'] = ', '.join(mcq_choices)

        return output_dict

    def convert_to_classification(self, processed_instance: Dict) -> Dict:

        processed_instance['input_sequence'] = self.classification_template.format(
            processed_instance.pop('question'), processed_instance.pop('possible_answers')
        )

        return processed_instance

    def convert_to_entailment(self, processed_instance: Dict) -> Dict:

        if self.dont_add_extra_text:
            hyp_str = processed_instance.pop('possible_answers')
            premise_str = processed_instance.pop('question')

        else:
            hyp_str = self.hypothesis_template.format(
                processed_instance.pop('possible_answers')
            )
            premise_str = self.premise_template.format(
                processed_instance.pop('question')
            )

        processed_instance['hypothesis'] = hyp_str
        processed_instance['premise'] = premise_str
        return processed_instance

    def convert_to_qa(self, processed_instance: Dict) -> Dict:
        processed_instance['question'] = processed_instance.pop('question')
        if self.dont_add_extra_text:
            context_str = processed_instance.pop('possible_answers')
        else:
            context_str = self.context_template.format(
                processed_instance.pop('possible_answers')
            )
        processed_instance['context'] = context_str
        return processed_instance
