

import configuration

from dhnamlib.pylib.context import Environment, LazyEval


config = Environment(
    evaluating_test_set=True,
    encoded_train_set=LazyEval(lambda: configuration.config.encoded_resplit_train_set),
    encoded_val_set=LazyEval(lambda: configuration.config.encoded_resplit_val_set),
    encoded_test_set=LazyEval(lambda: configuration.config.encoded_resplit_test_set),
)
