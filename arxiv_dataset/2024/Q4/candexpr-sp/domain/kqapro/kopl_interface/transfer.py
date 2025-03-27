
"Transferring from KoPL to Actions"

from itertools import chain
from collections import deque
# from functools import lru_cache
from functools import cache

# from configuration import config
from splogic.utility.trie import SequenceTrie
from splogic.seq2seq import learning
from splogic.seq2seq.token_pattern import is_bart_digit_seq_token

from dhnamlib.pylib.context import block
from dhnamlib.pylib.decoration import construct, curry, variable, id_cache, deprecated
from dhnamlib.pylib.function import compose
from dhnamlib.pylib.iteration import distinct_pairs, unique, merge_pairs, finditer, flatten

from . import read as kopl_read
from . import kb_analysis as kopl_kb_analysis


with block:
    # Processing token actions
    def is_digit_seq_token(token_value):
        return is_bart_digit_seq_token(token_value)

    def is_special_quantity_token(token_value):
        # Actually ['Ġ+', '+'] are not used to represent quantity in kqapro dataset
        return token_value in {'.', 'Ġ-', 'e', '-', 'Ġ+', '+'}

    def is_special_date_token(token_value):
        # return token_value in {'/'}  # '/' is used in kb.json but not in kopl dataset
        return token_value in {'-'}

    def is_special_year_token(token_value):
        # Actually 'Ġ+' is not used to represent year in kqapro dataset
        return token_value in {'Ġ-', 'Ġ+'}

    default_nl_token_union_type = (
        'kp-concept', 'kp-entity', 'kp-relation',
        'kp-attr-string', 'kp-attr-number', 'kp-attr-time',
        'kp-q-string', 'kp-q-number', 'kp-q-time',
        'vp-string', 'vp-unit')

    def get_token_act_type(token_value):
        is_digit_seq = is_digit_seq_token(token_value)
        is_special_quantity = is_special_quantity_token(token_value)
        is_special_year = is_special_year_token(token_value)
        is_special_date = is_special_date_token(token_value)

        union_type = list(default_nl_token_union_type)

        if is_digit_seq or is_special_quantity:
            union_type.append('vp-quantity')
        if is_digit_seq or is_special_date:
            union_type.append('vp-date')
        if is_digit_seq or is_special_year:
            union_type.append('vp-year')

        return tuple(union_type)

    data_type_to_attr_act_type = dict(
        string='kw-attr-string',
        quantity='kw-attr-number',
        time='kw-attr-time',
        date='kw-attr-time',
        year='kw-attr-time')
    data_type_to_q_act_type = dict(
        string='kw-q-string',
        quantity='kw-q-number',
        time='kw-q-time',
        date='kw-q-time',
        year='kw-q-time')

    def iter_act_type_trie_pairs(*, lf_tokenizer, end_of_seq, context):
        kb_info = kopl_kb_analysis.extract_kb_info(context.raw_kb)
        # data_types = set(['string', 'quantity', 'time', 'date', 'year'])
        # time_types = set(['time', 'date', 'year'])

        key_to_act_type = dict(
            concepts='kw-concept',
            # entities='kw-entity',
            relations='kw-relation',
            units='v-unit'
        )

        trie_dict = kopl_kb_analysis.make_trie_dict(kb_info, lf_tokenizer, end_of_seq)
        trie_dict['units'].add_token_seq([end_of_seq])  # add reduce-only case

        # Processing keywords in `key_to_act_type`
        for key, value in trie_dict.items():
            if isinstance(value, dict):
                pass
            elif key == 'entities':
                pass
            else:
                trie = value
                yield key_to_act_type[key], trie

        # Processing the trie for kw-entity
        entity_names = set()
        for entity_name, entities in context.kb.name_to_id.items():
            if len(entities) > 0:
                entity_names.add(entity_name)
        entity_trie = kopl_kb_analysis.make_trie(entity_names, lf_tokenizer, end_of_seq)
        yield 'kw-entity', entity_trie

        assert sum(1 for value in trie_dict.values() if isinstance(value, dict)) == 2

        # Processing keywords for attributes and qualifiers
        def merge_tries(tries):
            if len(tries) > 1:
                return SequenceTrie.merge(tries)
            else:
                return unique(tries)

        @variable
        @construct(curry(merge_pairs, merge_fn=merge_tries))
        def act_type_to_trie_dict():
            for data_type_to_act_type, data_type_to_trie in [[data_type_to_attr_act_type, trie_dict['type_to_attributes']],
                                                             [data_type_to_q_act_type, trie_dict['type_to_qualifiers']]]:
                for typ, trie in data_type_to_trie.items():
                    yield data_type_to_act_type[typ], trie

        yield from act_type_to_trie_dict.items()

    @id_cache
    def make_kw_to_type_dict(kb):
        kb_info = kopl_kb_analysis.extract_kb_info(kb)
        _kw_to_type_dict = kopl_kb_analysis.make_kw_to_type_dict(kb_info)

        @construct(dict)
        def _make_kw_to_type_dict(kw_category, new_type_dict):
            for kw, typ in _kw_to_type_dict[kw_category].items():
                yield kw, new_type_dict[typ]

        return dict(
            [kw_category, _make_kw_to_type_dict(kw_category, new_type_dict)]
            for kw_category, new_type_dict in [['attribute', data_type_to_attr_act_type],
                                               ['qualifier', data_type_to_q_act_type]])


with block:
    def kopl_to_action_tree(grammar, context, labeled_kopl_program):
        recursive_kopl_form = kopl_read.kopl_to_recursive_form(labeled_kopl_program)

        def parse(form):
            function_action = _kopl_function_to_action(grammar, form['function'])
            # yield function_action
            prev_kopl_input = None
            sub_trees = []
            for idx, kopl_input in enumerate(form['inputs']):
                sub_tree = _kopl_input_to_action_seq_or_action(
                    grammar, context, kopl_input, function_action.param_types[idx], prev_kopl_input)
                sub_trees.append(sub_tree)
                prev_kopl_input = kopl_input
            for sub_form in form['dependencies']:
                sub_trees.append(parse(sub_form))

            if len(sub_trees) == 0:
                return function_action
            else:
                return tuple([function_action] + sub_trees)

        return parse(recursive_kopl_form)

    def kopl_to_action_seq(grammar, context, labeled_kopl_program):
        return flatten(kopl_to_action_tree(grammar, context, labeled_kopl_program))

    # def kopl_to_action_seq(grammar, context, labeled_kopl_program):
    #     action_seq = []
    #     recursive_kopl_form = kopl_read.kopl_to_recursive_form(labeled_kopl_program)

    #     def parse(form):
    #         function_action = _kopl_function_to_action(grammar, form['function'])
    #         action_seq.append(function_action)
    #         prev_kopl_input = None
    #         for idx, kopl_input in enumerate(form['inputs']):
    #             action_seq.extend(_kopl_input_to_action_seq_or_action(
    #                 grammar, context, kopl_input, function_action.param_types[idx], prev_kopl_input))
    #             prev_kopl_input = kopl_input
    #         for sub_form in form['dependencies']:
    #             parse(sub_form)

    #     parse(recursive_kopl_form)
    #     return action_seq

    def _kopl_function_to_action(grammar, kopl_function):
        if kopl_function == 'What':
            kopl_function = 'QueryName'
        return _get_kopl_function_to_action_dict(grammar)[kopl_function]

    @cache
    @construct(compose(dict, distinct_pairs))
    def _get_kopl_function_to_action_dict(grammar):
        # kopl_function_to_action_dict = {}
        for action in grammar.base_actions:
            action_expr = action.expr_dict[grammar.formalism.default_expr_key]
            if isinstance(action_expr, str):
                kopl_function = kopl_read.extract_kopl_func(action_expr)
                if kopl_function is not None:
                    yield kopl_function, action

    def _kopl_input_to_action_seq_or_action(grammar, context, kopl_input, act_type, prev_kopl_input):
        def text_to_nl_token_actions(text):
            return tuple(
                chain(
                    map(compose(grammar.name_to_action, grammar.action_name_style.nl_token_to_action_name),
                        grammar.lf_tokenizer.tokenize(text)),
                    [grammar.reduce_action]))

        def is_type_of(act_type, super_type):
            return grammar.sub_and_super(act_type, super_type)

        output_action_seq = None
        output_action = None

        kw_to_type_dict = make_kw_to_type_dict(context.raw_kb)

        if is_type_of(act_type, 'keyword'):
            if act_type in ['kw-attr-comparable', 'kw-attribute']:
                new_act_type = kw_to_type_dict['attribute'][kopl_input]
                if act_type == 'kw-attr-comparable':
                    assert new_act_type in ['kw-attr-number', 'kw-attr-time']
            elif act_type == 'kw-qualifier':
                new_act_type = kw_to_type_dict['qualifier'][kopl_input]
            else:
                new_act_type = act_type
            keyword_action = unique(_act_type_to_actions(grammar, new_act_type))
            output_action_seq = tuple(chain([keyword_action], text_to_nl_token_actions(kopl_input)))
        elif is_type_of(act_type, 'operator'):
            candidate_actions = _act_type_to_actions(grammar, act_type)
            if kopl_input == '=':
                operator_action = unique(action for action in candidate_actions if action.name == 'op-eq')
            else:
                operator_action = unique(finditer(
                    candidate_actions, kopl_input,
                    test=lambda action, kopl_input: (kopl_input in action.expr_dict[grammar.formalism.default_expr_key])))
            # output_action_seq = [operator_action]
            output_action = operator_action
        elif is_type_of(act_type, 'value'):
            if act_type == 'value':
                new_act_type = 'v-{}'.format(kopl_read.classify_value_type(context, prev_kopl_input, kopl_input))
            else:
                new_act_type = act_type
            value_action = unique(_act_type_to_actions(grammar, new_act_type))
            output_action_seq = [value_action]
            if is_type_of(new_act_type, 'v-number'):
                assert value_action.name == 'constant-number'
                quantity_part, unit_part = kopl_read.number_to_quantity_and_unit(kopl_input)
                # quantity
                output_action_seq.append(grammar.name_to_action('constant-quantity'))
                output_action_seq.extend(text_to_nl_token_actions(quantity_part))
                # unit
                output_action_seq.append(grammar.name_to_action('constant-unit'))
                if unit_part == '1':
                    output_action_seq.extend([grammar.reduce_action])
                else:
                    output_action_seq.extend(text_to_nl_token_actions(unit_part))
            else:
                output_action_seq.extend(text_to_nl_token_actions(kopl_input))
        else:
            raise Exception(f'unexpected type "{act_type}"')

        if output_action_seq is not None:
            assert output_action is None
            return output_action_seq
        else:
            return output_action

    @cache
    def _get_act_type_to_actions_dict(grammar):
        return merge_pairs([action.act_type, action] for action in grammar.base_actions)

    def _act_type_to_actions(grammar, act_type):
        return _get_act_type_to_actions_dict(grammar)[act_type]


with block:
    @deprecated
    def iter_super_to_sub_actions(super_types_dict, is_non_conceptual_type):
        from splogic.base.formalism import Action

        super_sub_pair_set = set()

        def find_super_to_sub_actions(sub_type, super_types):
            for super_type in super_types:
                if is_non_conceptual_type(super_type):
                    super_sub_pair_set.add((super_type, sub_type))
                    find_super_to_sub_actions(super_type, super_types_dict.get(super_type, []))
                else:
                    find_super_to_sub_actions(sub_type, super_types_dict.get(super_type, []))

        for sub_type, super_types in super_types_dict.items():
            if is_non_conceptual_type(sub_type):
                find_super_to_sub_actions(sub_type, super_types)

        super_sub_pairs = sorted(super_sub_pair_set)
        for super_type, sub_type in super_sub_pairs:
            yield Action(name=f'{super_type}-to-{sub_type}',
                         act_type=super_type,
                         param_types=[sub_type],
                         expr_dict=dict(default='{0}'))

    @deprecated
    def get_strictly_typed_action_seq(grammar, action_name_seq, question):
        assert grammar.inferencing_subtypes is False

        input_action_seq = tuple(map(grammar.name_to_action, action_name_seq))
        num_processed_actions = 0
        output_action_seq = []

        def find_super_to_sub_type_seq(super_type, sub_type):
            type_seq_q = deque([typ] for typ in grammar.super_types_dict[sub_type])
            while type_seq_q:
                type_seq = type_seq_q.popleft()
                last_type = type_seq[-1]
                if super_type == last_type:
                    _type_seq = list(reversed([sub_type] + type_seq))
                    assert grammar.is_non_conceptual_type(_type_seq[0])
                    assert grammar.is_non_conceptual_type(_type_seq[-1])
                    type_seq = tuple(chain(
                        [_type_seq[0]],
                        filter(grammar.is_non_conceptual_type, _type_seq[1: -1]),
                        [_type_seq[-1]]))
                    return type_seq
                else:
                    type_seq_q.extend(type_seq + [typ] for typ in grammar.super_types_dict[last_type])
            else:
                raise Exception('Cannot find the type sequence')

        state = grammar.search_state_cls.create()
        while not state.tree.is_closed_root():
            expected_action = input_action_seq[num_processed_actions]
            num_processed_actions += 1
            utterance_token_id_seq = grammar.utterance_tokenizer(question)['input_ids']
            utterance_span_trie = learning.utterance_token_id_seq_to_span_trie(grammar, utterance_token_id_seq)
            with grammar.dynamic_scope.let(utterance_span_trie=utterance_span_trie):
                candidate_action_ids = state.get_candidate_action_ids()
            if expected_action.id in candidate_action_ids:
                output_action_seq.append(expected_action)
                state = state.get_next_state(expected_action)
            else:
                opened_tree, children = state.tree.get_opened_tree_children()
                opened_action = opened_tree.value
                param_type = opened_action.param_types[grammar.formalism._get_next_param_idx(opened_action, len(children))]
                type_seq = find_super_to_sub_type_seq(param_type, expected_action.act_type)

                for idx in range(len(type_seq) - 1):
                    lhs_type = type_seq[idx]
                    rhs_type = type_seq[idx + 1]
                    action_name = f'{lhs_type}-to-{rhs_type}'
                    intermediate_action = grammar.name_to_action(action_name)
                    state = state.get_next_state(intermediate_action)
                    output_action_seq.append(intermediate_action)
                output_action_seq.append(expected_action)
                state = state.get_next_state(expected_action)

        return output_action_seq
