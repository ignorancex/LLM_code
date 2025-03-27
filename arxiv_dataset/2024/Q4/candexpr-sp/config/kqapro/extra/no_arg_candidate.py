
from dhnamlib.pylib.context import Environment
from dhnamlib.pylib.context import LazyEval

from config.common.extra.no_arg_candidate import update_grammar, get_using_arg_candidate
from domain.kqapro import configuration


_action_names_for_arg_candidate = set([
    'keyword-concept',
    'keyword-entity',
    'keyword-relation',
    'keyword-attribute-string',
    'keyword-attribute-number',
    'keyword-attribute-time',
    'keyword-qualifier-string',
    'keyword-qualifier-number',
    'keyword-qualifier-time',
    'constant-unit',
])


config = Environment(
    grammar=LazyEval(lambda: update_grammar(configuration._make_grammar(), _action_names_for_arg_candidate)),
    ignoring_parsing_errors=True,
    using_arg_candidate=LazyEval(get_using_arg_candidate),
)
