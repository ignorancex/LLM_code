
import os
import json
from itertools import chain

from transformers import BartTokenizer

from splogic.seq2seq.transfer import ActionNameStyle
from splogic.utility.trie import SequenceTrie

from .data_format import parse_expr
from .utility import from_24_to_12, from_12_to_24, num_to_month, month_to_num


# OVERNIGHT_DOMAINS = ('basketball', 'blocks', 'calendar', 'housing', 'publications', 'recipes', 'restaurants', 'socialnetwork')
OVERNIGHT_DOMAINS = ('calendar', 'blocks', 'housing', 'restaurants', 'publications', 'recipes', 'socialnetwork', 'basketball')

_DEFAULT_NL_TOKEN_META_NAME = 'nl-token'
_DEFAULT_NL_TOKEN_META_ARG_NAME = 'token'
_REDUCE_ACTION_NAME = 'reduce'

_ACTION_NAME_STYLE = ActionNameStyle(_DEFAULT_NL_TOKEN_META_NAME)


def load_schema_info():
    schema_path = os.path.join(os.path.dirname(__file__), 'schema.json')
    with open(schema_path) as f:
        schema_info = json.load(f)
    return schema_info


SCHEMA_INFO = load_schema_info()


def make_relation_action_info():
    relation_action_info = {}
    for domain, schema in SCHEMA_INFO.items():
        relation_action_info[domain] = {}
        for category, exprs in schema.items():
            if category.startswith('relation'):
                if category in ['relation_number', 'relation_date', 'relation_time']:
                    action = 'keyword-relation-numeric'
                elif category == 'relation_entity':
                    action = 'keyword-relation-entity'
                elif category == 'relation_bool':
                    action = 'keyword-relation-bool'
                else:
                    raise Exception
                for expr in exprs:
                    relation_action_info[domain][expr] = action
    return relation_action_info


OP_ACTION_INFO = {
    "="  : 'op-eq',
    "<"  : 'op-lt',
    ">"  : 'op-gt',
    "<=" : 'op-le',
    ">=" : 'op-ge',
    "!=" : 'op-ne',
    "min": 'op-min',
    "max": 'op-max',
    "avg": 'op-avg',
    "sum": 'op-sum'
}


def typo_to_fixed(domain, expr):
    if domain == 'socialnetwork' and expr == 'en.city.bejing':
        return 'en.city.beijing'
    else:
        return expr


def fixed_to_typo(domain, expr):
    if domain == 'socialnetwork' and expr == 'en.city.beijing':
        return 'en.city.bejing'
    else:
        return expr


def labeled_logical_form_to_action_name_tree(domain, lf_tokenizer, action_name_style, labeled_logical_form):
    RELATION_ACTION_INFO = make_relation_action_info()

    parse = parse_expr(labeled_logical_form)

    # action_names = []

    # def add_action_name(action_name):
    #     action_names.append(action_name)

    # def _add_nl_token_action_names(nl_expr):
    #     for nl_token in lf_tokenizer.tokenize(nl_expr):
    #         add_action_name(action_name_style.nl_token_to_action_name(nl_token))

    # def add_nl_token_action_names(parent_action_name, nl_expr):
    #     add_action_name(parent_action_name)
    #     for nl_token in lf_tokenizer.tokenize(nl_expr):
    #         add_action_name(action_name_style.nl_token_to_action_name(nl_token))
    #     add_action_name(_REDUCE_ACTION_NAME)

    # def process_relation(relation_name_expr):
    #     add_nl_token_action_names(RELATION_ACTION_INFO[relation_name_expr], relation_name_expr)

    def get_nl_token_action_name_seq(nl_expr):
        return tuple(map(action_name_style.nl_token_to_action_name, lf_tokenizer.tokenize(nl_expr)))

    def get_nl_token_action_name_tree(parent_action_name, nl_expr):
        return (parent_action_name,) + get_nl_token_action_name_seq(nl_expr.replace('_', ' ')) + ('reduce',)

    def process_expr(parse):
        if isinstance(parse, str):
            assert parse.startswith('en.')
            expr_splits = typo_to_fixed(domain, parse).split('.')
            if len(expr_splits) == 2:
                return get_nl_token_action_name_tree('keyword-ent-type', expr_splits[-1])
            else:
                assert len(expr_splits) == 3
                return get_nl_token_action_name_tree('keyword-entity', expr_splits[-1])
        elif parse[0] == 'call':
            fn_expr = parse[1]
            arg_exprs = parse[2:]
            if fn_expr == 'SW.listValue':
                assert len(arg_exprs) == 1
                return ('program', process_expr(arg_exprs[0]))
            elif fn_expr == 'SW.getProperty':
                assert len(arg_exprs) == 2
                relation_expr = arg_exprs[1]
                if relation_expr == ('string', '!', 'type'):
                    entity_type_expr = arg_exprs[0]
                    return ('get-all-instances', process_expr(entity_type_expr))
                else:
                    entity_list_expr = arg_exprs[0]
                    return ('get-property', process_expr(relation_expr), process_expr(entity_list_expr))
                # elif relation_expr[:2] == ('call', 'SW.reverse'):
                #     entity_list_expr = arg_exprs[0]
                #     assert len(relation_expr) == 3
                #     assert relation_expr[2][0] == 'string'
                #     positive_relation_expr = relation_expr[2]
                #     return ('backward', process_expr(positive_relation_expr), process_expr(entity_list_expr))
                # elif relation_expr[0] == 'string':
                #     entity_list_expr = arg_exprs[0]
                #     assert len(relation_expr) == 2
                #     return ('forward', process_expr(relation_expr), process_expr(entity_list_expr))
                # else:
                #     breakpoint()
                #     raise Exception
            elif fn_expr == 'SW.reverse':
                assert len(arg_exprs) == 1
                relation_expr = arg_exprs[0]
                return ('reverse', process_expr(relation_expr))
            elif fn_expr == 'SW.domain':
                assert len(arg_exprs) == 1
                assert arg_exprs[0][0] == 'string'
                relation_expr = arg_exprs[0]
                return ('get-domain', process_expr(relation_expr))
            elif fn_expr == 'SW.concat':
                assert len(arg_exprs) == 2
                return ('concat', process_expr(arg_exprs[0]), process_expr(arg_exprs[1]))
                raise NotImplementedError
            elif fn_expr == 'SW.filter':
                if len(arg_exprs) == 2:
                    return ('filter-bool', process_expr(arg_exprs[1]), process_expr(arg_exprs[0]))
                else:
                    assert len(arg_exprs) == 4
                    return ('filter-comp',
                            process_expr(arg_exprs[1]),
                            process_expr(arg_exprs[2]),
                            process_expr(arg_exprs[3]),
                            process_expr(arg_exprs[0]))
            elif fn_expr == 'SW.superlative':
                assert len(arg_exprs) == 3
                return ('superlative',
                        process_expr(arg_exprs[2]),
                        process_expr(arg_exprs[1]),
                        process_expr(arg_exprs[0]))
            elif fn_expr == 'SW.countSuperlative':
                assert len(arg_exprs) in [3, 4]
                if len(arg_exprs) == 3:
                    optional_action_name_tree = 'reduce'
                else:
                    optional_action_name_tree = process_expr(arg_exprs[3])
                return ('count-superlative',
                        process_expr(arg_exprs[2]),
                        process_expr(arg_exprs[1]),
                        process_expr(arg_exprs[0]),
                        optional_action_name_tree)
            elif fn_expr == 'SW.countComparative':
                assert len(arg_exprs) in [4, 5]
                if len(arg_exprs) == 4:
                    optional_action_name_tree = 'reduce'
                else:
                    optional_action_name_tree = process_expr(arg_exprs[4])
                return ('count-comparative',
                        process_expr(arg_exprs[1]),
                        process_expr(arg_exprs[2]),
                        process_expr(arg_exprs[3]),
                        process_expr(arg_exprs[0]),
                        optional_action_name_tree)
            elif fn_expr == 'SW.aggregate':
                assert len(arg_exprs) == 2
                return ('aggregate', process_expr(arg_exprs[0]), process_expr(arg_exprs[1]))
            elif fn_expr in ['SW.singleton', 'SW.ensureNumericProperty', 'SW.ensureNumericEntity']:
                assert len(arg_exprs) == 1
                return process_expr(arg_exprs[0])
            elif fn_expr == '.size':
                assert len(arg_exprs) == 1
                return ('get-size', process_expr(arg_exprs[0]))
            else:
                breakpoint()
                raise Exception
        elif parse[0] == 'string':
            arg_exprs = parse[1:]
            if len(arg_exprs) != 1:
                assert arg_exprs == ('!', '=')
                arg_value = '!='
            else:
                arg_value = arg_exprs[0]
            if arg_value in ['=', '!=', '<', '>', '<=', '>=', 'min', 'max', 'avg', 'sum']:
                return OP_ACTION_INFO[arg_value]
            else:
                return get_nl_token_action_name_tree(RELATION_ACTION_INFO[domain][arg_value], arg_value)
        elif parse[0] == 'date':
            arg_exprs = parse[1:]
            year_expr, month_expr, day_expr = arg_exprs
            year_num, month_num, day_num = map(int, arg_exprs)
            if year_num == 2015:
                year_action_name_tree = 'constant-default-year'
            else:
                year_action_name_tree = get_nl_token_action_name_tree('constant-year', year_expr)
            if month_num == -1:
                assert day_num == -1
                return ('constant-date', year_action_name_tree, 'reduce')
            else:
                assert day_num != -1
                month_action_name_tree = get_nl_token_action_name_tree('constant-month', num_to_month(int(month_expr)))
                day_action_name_tree = get_nl_token_action_name_tree('constant-day', day_expr)
                return ('constant-date', year_action_name_tree, month_action_name_tree, day_action_name_tree)
        elif parse[0] == 'time':
            arg_exprs = parse[1:]
            hour_expr, minute_expr = arg_exprs
            hour_num, minute_num = map(int, arg_exprs)
            assert minute_num == 0
            new_hour, am_or_pm = from_24_to_12(hour_num)
            return ('constant-number',
                    get_nl_token_action_name_tree('constant-quantity', str(new_hour)),
                    get_nl_token_action_name_tree('constant-unit', am_or_pm))
        elif parse[0] == 'number':
            arg_exprs = parse[1:]
            if len(arg_exprs) == 1:
                quantity_expr = arg_exprs[0]
                return ('constant-number',
                        get_nl_token_action_name_tree('constant-quantity', quantity_expr),
                        'reduce')
            else:
                assert len(arg_exprs) == 2
                quantity_expr, unit_expr = arg_exprs
                if unit_expr.startswith('en.'):
                    unit_expr = unit_expr[len('en.'):]
                return ('constant-number',
                        get_nl_token_action_name_tree('constant-quantity', quantity_expr),
                        get_nl_token_action_name_tree('constant-unit', unit_expr))
        elif isinstance(parse[0], tuple):
            assert parse[0][0] == 'lambda'
            lambda_expr = parse[0]
            lambda_arg_expr = parse[1]
            lambda_body_expr = lambda_expr[2]
            return process_expr(replace_var(lambda_body_expr, lambda_arg_expr))
        else:
            breakpoint()
            raise NotImplementedError

    action_name_tree = process_expr(parse)

    return action_name_tree


def replace_var(expr, replacement):
    if expr == ('var', 's'):
        return replacement
    elif isinstance(expr, str):
        return expr
    else:
        return tuple(
            replace_var(sub_expr, replacement)
            for sub_expr in expr)


# Modification
# - (time {hour} 0) -> (number {number} am/pm)
# - underscore(_) is replaced with space( ) from the names of relation and entity names.
# - prefixs such as "en." or "en.{entity_type}" are removed from entity, entity_type, units.
# - a month is mapped to the name of the month rather than a number. (e.g. 1 -> january)


class ExprMapper:
    def __init__(self):
        self.ent_type_mapper = {}
        self.entity_mapper = {}
        self.unit_mapper = {}
        for domain, schema in SCHEMA_INFO.items():
            self.ent_type_mapper[domain] = {}
            self.entity_mapper[domain] = {}
            self.unit_mapper[domain] = {}
            for full_entity_expr in schema['entity']:
                en, brief_ent_type_expr, brief_entity_expr = full_entity_expr.split('.')
                full_ent_type_expr = f'{en}.{brief_ent_type_expr}'
                self.entity_mapper[domain][brief_entity_expr] = full_entity_expr
                self.ent_type_mapper[domain][brief_ent_type_expr] = full_ent_type_expr
            for full_ent_type_expr in schema.get('extra_type', []):
                brief_ent_type_expr = full_ent_type_expr.split('.')[-1]
                self.ent_type_mapper[domain][brief_ent_type_expr] = full_ent_type_expr
            for full_unit_expr in schema.get('number_unit', []):
                if full_unit_expr.startswith('en.'):
                    brief_unit_expr = full_unit_expr.split('.')[-1]
                else:
                    brief_unit_expr = full_unit_expr
                self.unit_mapper[domain][brief_unit_expr] = full_unit_expr

        self.unit_mapper['calendar']['am'] = 'am'
        self.unit_mapper['calendar']['pm'] = 'pm'

    @staticmethod
    def remove_space(text):
        return text.replace(' ', '_')

    def get_ent_type(self, domain, ent_type_expr):
        return self.ent_type_mapper[domain][self.remove_space(ent_type_expr)]

    def get_entity(self, domain, entity_expr):
        mapped_expr = self.entity_mapper[domain][self.remove_space(entity_expr)]
        return fixed_to_typo(domain, mapped_expr)

    def get_unit(self, domain, unit_expr):
        return self.unit_mapper[domain][self.remove_space(unit_expr)]

    @staticmethod
    def get_month(mont_expr):
        return str(month_to_num(mont_expr))

    @staticmethod
    def get_time(hour_expr, unit_expr):
        try:
            hour = from_12_to_24(int(hour_expr), unit_expr)
        except AssertionError:
            return 0
        else:
            return str(hour)


def make_trie_info(lf_tokenizer, end_of_seq):
    def make_trie():
        return SequenceTrie(tokenizer=lf_tokenizer, end_of_seq=end_of_seq)

    def add_text(trie, text):
        trie.add_text(text.replace('_', ' '))

    trie_info = {}

    month_trie = make_trie()
    for full_month_expr in map(num_to_month, range(1, 13)):
        add_text(month_trie, full_month_expr)

    day_trie = make_trie()
    for full_day_expr in map(str, range(1, 32)):
        add_text(day_trie, full_day_expr)

    for domain, schema in SCHEMA_INFO.items():
        trie_info[domain] = {}

        ent_type_trie = make_trie()
        entity_trie = make_trie()
        for full_entity_expr in schema['entity']:
            en, brief_ent_type_expr, brief_entity_expr = full_entity_expr.split('.')
            add_text(ent_type_trie, brief_ent_type_expr)
            add_text(entity_trie, brief_entity_expr)
        for full_ent_type_expr in schema.get('extra_type', []):
            brief_ent_type_expr = full_ent_type_expr.split('.')[-1]
            add_text(ent_type_trie, brief_ent_type_expr)
        trie_info[domain]['ent-type'] = ent_type_trie
        trie_info[domain]['entity'] = entity_trie

        relation_entity_trie = make_trie()
        for full_relation_entity_expr in schema['relation_entity']:
            add_text(relation_entity_trie, full_relation_entity_expr)
        trie_info[domain]['relation-entity'] = relation_entity_trie

        relation_bool_trie = make_trie()
        for full_relation_bool_expr in schema.get('relation_bool', []):
            add_text(relation_bool_trie, full_relation_bool_expr)
        trie_info[domain]['relation-bool'] = relation_bool_trie

        relation_numeric_trie = make_trie()
        for full_relation_numeric_expr in chain(
                schema.get('relation_number', []),
                schema.get('relation_date', []),
                schema.get('relation_time', []),
        ):
            add_text(relation_numeric_trie, full_relation_numeric_expr)
        trie_info[domain]['relation-numeric'] = relation_numeric_trie

        trie_info[domain]['month'] = month_trie
        trie_info[domain]['day'] = day_trie

        unit_trie = make_trie()
        for full_unit_expr in schema.get('number_unit', []):
            if full_unit_expr.startswith('en.'):
                brief_unit_expr = full_unit_expr.split('.')[-1]
            else:
                brief_unit_expr = full_unit_expr
            add_text(unit_trie, brief_unit_expr)
        if domain == 'calendar':
            add_text(unit_trie, 'am')
            add_text(unit_trie, 'pm')
        trie_info[domain]['unit'] = unit_trie

    return trie_info


__ACTS_USING_ARG_CANDIDATES = (
    'keyword-ent-type',
    'keyword-entity',
    'keyword-relation-entity',
    'keyword-relation-bool',
    'keyword-relation-numeric',
    'constant-month',
    'constant-day',
    'constant-unit',
)

ACT_TYPES_FOR_ARG_CANDIDATES = (
    'ent-type',
    'entity',
    'relation-entity',
    'relation-bool',
    'relation-numeric',
    'month',
    'day',
    'unit',
)


def test_one_example():
    lf_tokenizer = BartTokenizer.from_pretrained(
        'facebook/bart-base',
        add_prefix_space=True)

    domain = 'basketball'
    labeled_logical_form = '( call SW.listValue ( call SW.getProperty ( ( lambda s ( call SW.filter ( var s ) ( call SW.ensureNumericProperty ( string num_assists ) ) ( string < ) ( call SW.ensureNumericEntity ( number 3 assist ) ) ) ) ( call SW.domain ( string player ) ) ) ( string player ) ) )'
    action_name_tree = labeled_logical_form_to_action_name_tree(domain, lf_tokenizer, _ACTION_NAME_STYLE, labeled_logical_form)
    print(action_name_tree)


def test_examples():
    import pandas
    from dhnamlib.pylib import filesys

    lf_tokenizer = BartTokenizer.from_pretrained(
        'facebook/bart-base',
        add_prefix_space=True)
    dataset_dir_path = './dataset/overnight'
    processed_dir_path = './processed/overnight/action_name_tree'

    split_type = 'test'

    for domain in OVERNIGHT_DOMAINS:
        dataset_file_path = os.path.join(dataset_dir_path, f'{domain}_{split_type}.tsv')
        dataset_df = pandas.read_csv(dataset_file_path, sep='\t', header=0)

        processed_examples = []

        for idx, row in dataset_df.iterrows():
            utterance = row['utterance']
            action_name_tree = labeled_logical_form_to_action_name_tree(domain, lf_tokenizer, _ACTION_NAME_STYLE, row['logical_form'])
            # original = row['original']
            processed_examples.append((utterance, action_name_tree))

        processed_df = pandas.DataFrame(processed_examples, columns=['utterance', 'action_name_tree'])
        filesys.mkloc_unless_exist(processed_dir_path)
        processed_df.to_csv(os.path.join(processed_dir_path, f'{domain}_{split_type}.tsv'), sep='\t')
    pass


if __name__ == '__main__':
    test_examples()
