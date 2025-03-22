
from dhnamlib.pylib.context import Environment, LazyEval

from domain.kqapro.configuration import _make_ablation_grammar

import configuration

# ns: non-symbolic


config = Environment(
    grammar=LazyEval(lambda: _make_ablation_grammar(non_symbolic=True)),
    using_arg_candidate=False,
    constrained_decoding=False,

    encoded_train_set=LazyEval(lambda: configuration.config.encoded_ns_train_set),
    encoded_val_set=LazyEval(lambda: configuration.config.encoded_ns_val_set),

    shuffled_encoded_train_set=LazyEval(lambda: configuration.config.shuffled_encoded_ns_train_set),
    shuffled_encoded_val_set=LazyEval(lambda: configuration.config.shuffled_encoded_ns_val_set),
)
