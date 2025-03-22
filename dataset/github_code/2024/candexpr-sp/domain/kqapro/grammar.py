
from functools import cache

# from dhnamlib.pylib.klass import Interface
from dhnamlib.pylib.klass import subclass, implement, override
from dhnamlib.pylib.context import Environment
from dhnamlib.pylib.debug import NIE
from dhnamlib.hissplib.expression import repr_as_hash_str
from dhnamlib.pylib.lazy import DynamicLazyProxy
from splogic.utility.trie import NoTrie

from splogic.seq2seq.grammar import Seq2SeqGrammar
from splogic.utility.trie import MergedTrie
from splogic.seq2seq.transfer import ActionNameStyle, StrictTypeProcessing

from configuration import config

# from .execution import KoPLCompiler
from .kopl_interface import kb_analysis as kopl_kb_analysis
from .kopl_interface import transfer as kopl_transfer
from .transfer import KQAProTokenProcessing


_DEFAULT_NL_TOKEN_META_NAME = 'nl-token'
_DEFAULT_NL_TOKEN_META_ARG_NAME = 'token'


@subclass
class KQAProGrammar(Seq2SeqGrammar):
    # interface = Interface(Seq2SeqGrammar)

    @config
    def __init__(
            self, *,
            #
            formalism, super_types_dict, actions, start_action, meta_actions, register,
            is_non_conceptual_type=None, use_reduce=True,
            inferencing_subtypes=config.ph(True),
            #
            using_distinctive_union_types=config.ph(True),
            #
            # using_spans_as_entities=config.ph(False),
            pretrained_model_name_or_path=config.ph
    ):
        # dynamic_scope = Environment(allowing_duplicate_candidate_ids=True)
        dynamic_scope = Environment(utterance_span_trie=NoTrie())
        token_processing = KQAProTokenProcessing()
        action_name_style = ActionNameStyle(_DEFAULT_NL_TOKEN_META_NAME)
        strict_type_processing = StrictTypeProcessing()

        super().__init__(
            formalism=formalism, super_types_dict=super_types_dict, actions=actions, start_action=start_action,
            meta_actions=meta_actions, register=register,
            is_non_conceptual_type=is_non_conceptual_type, use_reduce=use_reduce,
            inferencing_subtypes=inferencing_subtypes,
            #
            dynamic_scope=dynamic_scope,
            #
            using_distinctive_union_types=using_distinctive_union_types,
            #
            token_processing=token_processing,
            action_name_style=action_name_style,
            strict_type_processing=strict_type_processing,
            #
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            #
            nl_token_meta_name=_DEFAULT_NL_TOKEN_META_NAME,
            nl_token_meta_arg_name=_DEFAULT_NL_TOKEN_META_ARG_NAME,
        )

    @config
    @cache
    @override
    def get_search_state_cls(self, using_arg_candidate=config.ph, using_arg_filter=config.ph):
        return super().get_search_state_cls(
            using_arg_candidate=using_arg_candidate,
            using_arg_filter=using_arg_filter
        )

    # @implement
    # def get_compiler_cls(self):
    #     return KoPLCompiler

    @config
    @implement
    def register_specific(self, register, global_context=config.ph):
        @register('(function concat-parts)')
        def concat_parts(*tokens):
            return repr_as_hash_str(self.join_nl_tokens(tokens))

        @register('(function concat-quantity-unit)')
        def concat_quantity_unit(quantity, unit):
            # Since Lissp's raw string cannot express double quote("),
            # hash string should be used.
            return repr_as_hash_str(f'{quantity} {unit}'.rstrip())

        def get_act_type(typ):
            return f'v-{typ}'

        def make_is_valid_expr(typ):
            def is_valid_expr(expr):
                return kopl_kb_analysis.is_value_type(expr, typ)

            return is_valid_expr

        typ_is_prefix_pairs = [['quantity', kopl_kb_analysis.is_quantity_prefix],
                               ['date', kopl_kb_analysis.is_date_prefix],
                               ['year', kopl_kb_analysis.is_year_prefix]]

        for typ, is_valid_prefix in typ_is_prefix_pairs:
            register(f'(filter {get_act_type(typ)})',
                     self.make_arg_filter(is_valid_prefix, make_is_valid_expr(typ)))

        for act_type, trie in kopl_transfer.iter_act_type_trie_pairs(
                lf_tokenizer=self.lf_tokenizer, end_of_seq=self.reduce_token, context=global_context
        ):
            if act_type == 'kw-entity':
                trie.ignoring_errors = True
                arg_candidate = self.make_trie_arg_candidate(MergedTrie(
                    [trie, DynamicLazyProxy(lambda: self.dynamic_scope.utterance_span_trie)],
                    allowing_duplicates=True))
            else:
                arg_candidate = self.make_trie_arg_candidate(trie)
            register(f'(candidate {act_type})', arg_candidate)


# TODO
# - dynamic_let vs. let_dynamic_trie
