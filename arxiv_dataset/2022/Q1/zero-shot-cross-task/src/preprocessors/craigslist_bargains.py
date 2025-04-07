from copy import deepcopy
from typing import Dict
import math

from src.preprocessors.task_preprocessor import FixedChoiceTaskPreprocessor


@FixedChoiceTaskPreprocessor.register('craigslist')
class CraigslistBargainsPreprocessor(FixedChoiceTaskPreprocessor):

    def __init__(
            self,
            add_speaker_prefix: bool = False,
            fair_trade_tol=1e-3,
            choices=None,
            classification_template: str = None,
            premise_template: str = None,
            hypothesis_template: str = None,
            question_template: str = None,
            context_template: str = None,
            use_constant_domain: bool = False,
            is_mcq: bool = False,
            choice_str: str = None,
            mcq_choice_str: str = None,
            dont_add_extra_text: bool = False
    ):
        self.question_str = "Who won the negotiation?"

        self.background_info_str = "The buyer wanted ${:.2f}, the seller wanted" \
                                   " ${:.2f}, and the final price is ${:.2f}."

        if choices is None:
            choices = ["Seller", "Buyer", "Neither", "Unknown"]
        assert len(choices) == 4

        classification_template = (
                classification_template
                or self.background_info_str + " {}"
        )
        premise_template = (
                premise_template
                or "{}"
        )
        hypothesis_template = (
                hypothesis_template
                or self.background_info_str
        )
        context_template = (
                context_template
                or "{}"
        )
        question_template = (
                question_template
                or self.background_info_str + ' ' + self.question_str
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
        self.add_speaker_prefix = add_speaker_prefix
        self.fair_trade_tol = fair_trade_tol
        self.label_to_int = {
            "SELLER" : 0,
            "BUYER"  : 1,
            "NEITHER": 2,
            "UNKNOWN": 3
        }
        self.use_constant_domain = use_constant_domain

    def _process_instance(self, data_instance, idx) -> Dict:

        item_category = data_instance['items']['Category'][0]
        listing_price = data_instance['items']['Price'][0]
        output_dict = {
            "choices": self.choices,
            "idx"    : idx,
            "domain" : "negotiations" if self.use_constant_domain else item_category,
        }

        # Get the utterances
        input_sequence = []
        for i, utterance in enumerate(data_instance['utterance']):
            prefix = ""
            if self.add_speaker_prefix:
                # Buyer always speaks first then seller, so on the even values
                # of i, the buyer will be speaking and on the odd values the
                # seller will be speaking.
                prefix = "Buyer" if i % 2 == 0 else "Seller"
                prefix += ": "

            input_sequence.append(f"{prefix}{utterance}")

        output_dict['input_sequence'] = "\n\n".join(input_sequence)
        label, final_price, buyer_target, seller_target = self._get_label(
            data_instance['dialogue_acts']['intent'],
            data_instance['dialogue_acts']['price'],
            data_instance['agent_info']['Target']
        )
        output_dict['final_price'] = final_price
        output_dict['buyer_target'] = buyer_target
        output_dict['seller_target'] = seller_target
        output_dict['label'] = self.label_to_int[label]

        output_dict['additional_input_1'] = listing_price

        return output_dict

    def _get_label(self, intents, prices, targets):
        if not intents:
            return "UNKNOWN", -1, -1, -1

        final_price = -1
        # Get both parties target price.
        buyer_target, seller_target = targets
        found_intent = False
        for i, p in zip(intents[::-1], prices[::-1]):

            if i:
                found_intent = True
            if p > -1:
                final_price = p
                break

        # Use less than 0 as it is impossible to have less than zero price.
        if final_price < 0 or not found_intent:
            return "UNKNOWN", -1, buyer_target, seller_target

        if intents[-1] != 'accept':
            return "NEITHER", -1, buyer_target, seller_target

        # Get how far off the final price was from both parties targets
        buyer_diff = abs(final_price - buyer_target)
        seller_diff = abs(final_price - seller_target)

        # Dealing with floats, so use isclose for safety with tol
        if math.isclose(seller_diff, buyer_diff, rel_tol=self.fair_trade_tol):
            label = "NEITHER"
        elif seller_diff > buyer_diff:
            label = "BUYER"
        elif seller_diff < buyer_diff:
            label = "SELLER"
        else:
            raise ValueError("SOMETHING WENT VERY WRONG!")
        return label, final_price, buyer_target, seller_target

    def convert_to_classification(self, processed_instance: Dict) -> Dict:

        if not self.dont_add_extra_text:
            processed_instance['input_sequence'] = self.classification_template.format(
                processed_instance.pop('buyer_target'),
                processed_instance.pop('seller_target'),
                processed_instance.pop('final_price'),
                processed_instance['input_sequence']
            )
        return processed_instance

    def convert_to_entailment(self, processed_instance: Dict) -> Dict:
        hyp_str = ""
        premise_str = processed_instance.pop('input_sequence')
        if not self.dont_add_extra_text:
            hyp_str = self.hypothesis_template.format(
                processed_instance.pop('buyer_target'),
                processed_instance.pop('seller_target'),
                processed_instance.pop('final_price'),
            )
            premise_str = self.premise_template.format(premise_str)
        processed_instance['premise'] = premise_str
        processed_instance['hypothesis'] = hyp_str

        return processed_instance

    def convert_to_qa(self, processed_instance: Dict) -> Dict:

        question_str = ""
        context_str = processed_instance.pop('input_sequence')
        if not self.dont_add_extra_text:
            question_str = self.question_template.format(
                processed_instance.pop('buyer_target'),
                processed_instance.pop('seller_target'),
                processed_instance.pop('final_price'),
            )
            context_str = self.context_template.format(context_str)
        processed_instance['question'] = question_str
        processed_instance['context'] = context_str
        return processed_instance
