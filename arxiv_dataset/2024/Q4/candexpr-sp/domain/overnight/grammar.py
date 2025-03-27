
from functools import cache

# from dhnamlib.pylib.klass import Interface
from dhnamlib.pylib.klass import subclass, implement, override
from dhnamlib.pylib.context import Environment
from dhnamlib.pylib.debug import NIE
# from dhnamlib.hissplib.expression import repr_as_hash_str
from dhnamlib.pylib.lazy import DynamicLazyProxy
from splogic.utility.trie import NoTrie

# from splogic.base.execution import ExprCompiler
from splogic.seq2seq.grammar import Seq2SeqGrammar
from splogic.seq2seq.transfer import ActionNameStyle, StrictTypeProcessing
# from splogic.utility.trie import MergedTrie

from configuration import config

from .transter import OvernightTokenProcessing
from .lf_interface.transfer import ExprMapper, make_trie_info, ACT_TYPES_FOR_ARG_CANDIDATES


_DEFAULT_NL_TOKEN_META_NAME = 'nl-token'
_DEFAULT_NL_TOKEN_META_ARG_NAME = 'token'


@subclass
class OvernightGrammar(Seq2SeqGrammar):
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
        token_processing = OvernightTokenProcessing()
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
    #     return ExprCompiler

    @implement
    def register_specific(self, register):
        expr_mapper = ExprMapper()

        def default_join_tokens(tokens):
            return self.join_nl_tokens(tokens).replace(' ', '_')

        def default_concat_tokens(*tokens):
            return default_join_tokens(tokens)

        def visual_join_tokens(tokens):
            return self.join_nl_tokens(tokens).replace(' ', '-')

        def visual_concat_tokens(*tokens):
            return visual_join_tokens(tokens)

        @register('(default-expr-fn ent-type)')
        def get_default_expr_of_ent_type(*tokens):
            return expr_mapper.get_ent_type(self.dynamic_scope.domain, default_join_tokens(tokens))

        @register('(visual-expr-fn ent-type)')
        def get_visual_expr_of_ent_type(*tokens):
            return 'type:{}'.format(visual_join_tokens(tokens))

        @register('(default-expr-fn relation-entity)')
        @register('(default-expr-fn relation-bool)')
        @register('(default-expr-fn relation-numeric)')
        def get_default_expr_of_relation(*tokens):
            return '( string {} )'.format(default_join_tokens(tokens))

        @register('(visual-expr-fn relation-entity)')
        @register('(visual-expr-fn relation-bool)')
        @register('(visual-expr-fn relation-numeric)')
        def get_visual_expr_of_relation(*tokens):
            return 'relation:{}'.format(visual_join_tokens(tokens))

        @register('(default-expr-fn entity)')
        def get_default_expr_of_entity(*tokens):
            return expr_mapper.get_entity(self.dynamic_scope.domain, default_join_tokens(tokens))

        @register('(visual-expr-fn entity)')
        def get_visual_expr_of_entity(*tokens):
            return 'entity:{}'.format(visual_join_tokens(tokens))

        @register('(default-expr-fn month)')
        @register('(visual-expr-fn month)')
        def get_default_expr_of_month(*tokens):
            return expr_mapper.get_month(self.join_nl_tokens(tokens))

        @register('(default-expr-fn unit)')
        def get_default_expr_of_unit(*tokens):
            return expr_mapper.get_unit(self.dynamic_scope.domain, default_join_tokens(tokens))

        @register('(visual-expr-fn unit)')
        def get_visual_expr_of_unit(*tokens):
            return visual_join_tokens(tokens).replace(' ', '-')

        register('(default-expr-fn quantity)', self.concat_nl_tokens)
        register('(visual-expr-fn quantity)', self.concat_nl_tokens)
        register('(default-expr-fn year)', self.concat_nl_tokens)
        register('(visual-expr-fn year)', self.concat_nl_tokens)
        register('(default-expr-fn day)', self.concat_nl_tokens)
        register('(visual-expr-fn day)', self.concat_nl_tokens)

        @register('(default-expr-fn number)')
        def get_default_expr_of_number(quantity_expr, unit_expr):
            if unit_expr in ['am', 'pm']:
                return '( time {} 0 )'.format(expr_mapper.get_time(quantity_expr, unit_expr))
            else:
                if unit_expr == '':
                    return '( number {} )'.format(quantity_expr)
                else:
                    return '( number {} {} )'.format(quantity_expr, unit_expr)

        @register('(visual-expr-fn number)')
        def get_visual_expr_of_number(quantity_expr, unit_expr):
            return '{}~{}'.format(quantity_expr, unit_expr)

        @register('(default-expr-fn date)')
        def get_default_expr_of_date(year_expr, month_expr, day_expr=None):
            if month_expr == '':
                assert day_expr is None
                return '( date {} {} {} )'.format(year_expr, '-1', '-1')
            else:
                return '( date {} {} {} )'.format(year_expr, month_expr, day_expr)

        @register('(visual-expr-fn date)')
        def get_visual_expr_of_date(year_expr, month_expr, day_expr=None):
            if month_expr == '':
                assert day_expr is None
                return '-'.join((year_expr, 'xx', 'xx'))
            else:
                if len(month_expr) == 1:
                    month_expr = '0' + month_expr
                if len(day_expr) == 1:
                    day_expr = '0' + day_expr
                return '-'.join((year_expr, month_expr, day_expr))

        def make_trie_constructor(trie_info, dynamic_scope, act_type):
            return lambda: trie_info[dynamic_scope.domain][act_type]

        trie_info = make_trie_info(lf_tokenizer=self.lf_tokenizer, end_of_seq=self.reduce_token)
        for act_type in ACT_TYPES_FOR_ARG_CANDIDATES:
            # arg_candidate = self.make_trie_arg_candidate(
            #     DynamicLazyProxy(lambda: trie_info[self.dynamic_scope.domain][act_type]))
            # arg_candidate = self.make_trie_arg_candidate(
            #     DynamicLazyProxy(TrieConstructor(trie_info, self.dynamic_scope, act_type)))
            arg_candidate = self.make_trie_arg_candidate(
                DynamicLazyProxy(make_trie_constructor(trie_info, self.dynamic_scope, act_type)))
            register(f'(candidate {act_type})', arg_candidate)


# class TrieConstructor:
#     def __init__(self, trie_info, dynamic_scope, act_type):
#         self.trie_info = trie_info
#         self.dynamic_scope = dynamic_scope
#         self.act_type = act_type

#     def __call__(self):
#         return self.trie_info[self.dynamic_scope.domain][self.act_type]
