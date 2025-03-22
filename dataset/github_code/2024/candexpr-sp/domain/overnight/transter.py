
import itertools
import re

from dhnamlib.pylib.klass import subclass, implement
from dhnamlib.pylib.iteration import rmap

from splogic.seq2seq.transfer import TokenProcessing
from splogic.seq2seq.token_pattern import is_bart_digit_seq_token

# from .lf_interface import transfer as lf_transfer
from .lf_interface.transfer import labeled_logical_form_to_action_name_tree


@subclass
class OvernightTokenProcessing(TokenProcessing):

    default_nl_token_union_type = (
        'part-ent-type',
        'part-relation-entity',
        'part-relation-bool',
        'part-relation-numeric',
        'part-entity',
        # 'part-quantity',
        'part-unit',
        # 'part-year',
        'part-month',
        # 'part-day',
    )

    full_nl_token_union_type = \
        tuple(default_nl_token_union_type) + \
        ('part-quantity', 'part-year', 'part-day')

    @classmethod
    @implement
    def get_token_act_type(cls, token_value):
        if is_bart_digit_seq_token(token_value):
            return cls.full_nl_token_union_type
        else:
            return cls.default_nl_token_union_type

    @classmethod
    @implement
    def get_non_distinctive_nl_token_act_type(cls, token_value):
        return cls.full_nl_token_union_type

    # labeled_logical_form_to_action_seq = staticmethod(interface.implement(
    #     kopl_transfer.kopl_to_action_seq))

    # @staticmethod
    # @implement
    # def labeled_logical_form_to_action_seq(grammar, context, labeled_logical_form):
    #     raise NotImplementedError

    @staticmethod
    @implement
    def labeled_logical_form_to_action_tree(labeled_logical_form, grammar, domain):
        action_name_tree = labeled_logical_form_to_action_name_tree(
            domain=domain,
            lf_tokenizer=grammar.lf_tokenizer,
            action_name_style=grammar.action_name_style,
            labeled_logical_form=labeled_logical_form
        )

        return rmap(grammar.convert_name_to_action, action_name_tree)
