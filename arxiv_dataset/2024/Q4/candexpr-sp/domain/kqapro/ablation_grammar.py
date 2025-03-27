
# from dhnamlib.pylib.structure import LazyDict
from dhnamlib.pylib.lazy import LazyEval, LazyProxy


def is_symbolic_action(action):
    return action.name in symbolic_actions


symbolic_actions = set([
    'all-entities', 'find', 'filter-concept', 'filter-str', 'filter-number', 'filter-year', 'filter-date', 'relate',
    # 'op-eq', 'op-ne', 'op-lt', 'op-gt',
    'direction-forward', 'direction-backward',
    'q-filter-str', 'q-filter-number', 'q-filter-year', 'q-filter-date',
    'intersect', 'union', 'count', 'select-between', 'select-among',
    # 'op-st', 'op-bt', 'op-min', 'op-max'
    'query-name', 'query-attr', 'query-attr-under-cond', 'query-relation', 'query-attr-qualifier', 'query-rel-qualifier',
    'verify-str', 'verify-number', 'verify-year', 'verify-date'
])  # Names that consist of acronyms are excluded. (e.g. op-eq, op-st)


def is_nl_token_seq_action(action):
    return action.name in nl_token_seq_actions


nl_token_seq_actions = set([
    'keyword-concept',
    'keyword-entity',
    'keyword-relation',
    'keyword-attribute-string',
    'keyword-attribute-number',
    'keyword-attribute-time',
    'keyword-qualifier-string',
    'keyword-qualifier-number',
    'keyword-qualifier-time',
    'constant-string',
    # 'constant-quantity',
    'constant-year',
    'constant-date',
    # 'constant-number',
    # 'constant-unit',
])


def make_common_nl_token_seq_expr_dict(grammar_fn):
    def make_expr_dict():
        return grammar_fn().register.retrieve('(function concat-parts)')

    common_nl_token_seq_expr_dict = dict(
        default=LazyProxy(make_expr_dict)
    )
    return common_nl_token_seq_expr_dict
