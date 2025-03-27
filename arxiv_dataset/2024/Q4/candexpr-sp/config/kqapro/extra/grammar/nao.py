
from dhnamlib.pylib.context import Environment, LazyEval

from domain.kqapro.configuration import _make_ablation_grammar

import configuration

# nao: naive_arg_ordering


config = Environment(
    grammar=LazyEval(lambda: _make_ablation_grammar(naive_arg_ordering=True)),
    using_arg_candidate=False,
    constrained_decoding=False,

    encoded_train_set=LazyEval(lambda: configuration.config.encoded_nao_train_set),
    encoded_val_set=LazyEval(lambda: configuration.config.encoded_nao_val_set),

    shuffled_encoded_train_set=LazyEval(lambda: configuration.config.shuffled_encoded_nao_train_set),
    shuffled_encoded_val_set=LazyEval(lambda: configuration.config.shuffled_encoded_nao_val_set),
)
