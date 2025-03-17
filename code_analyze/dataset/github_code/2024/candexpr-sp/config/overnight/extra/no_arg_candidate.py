
from dhnamlib.pylib.context import Environment
from dhnamlib.pylib.context import LazyEval

from config.common.extra.no_arg_candidate import update_grammar, get_using_arg_candidate
from domain.overnight import configuration


_action_names_for_arg_candidate = set([
    'keyword-ent-type',
    'keyword-relation-entity',
    'keyword-relation-bool',
    'keyword-relation-numeric',
    'keyword-entity',
    'constant-month',
    'constant-day',
    'constant-unit'
])


config = Environment(
    grammar=LazyEval(lambda: update_grammar(configuration._make_grammar(), _action_names_for_arg_candidate)),
    ignoring_parsing_errors=True,
    using_arg_candidate=LazyEval(get_using_arg_candidate),
)
