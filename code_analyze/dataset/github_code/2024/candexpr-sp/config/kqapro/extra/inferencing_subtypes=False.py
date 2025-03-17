import configuration

from dhnamlib.pylib.context import Environment
from dhnamlib.pylib.context import LazyEval

config = Environment(
    inferencing_subtypes=False,
    encoded_train_set=LazyEval(lambda: configuration.config.encoded_strict_train_set),
    encoded_val_set=LazyEval(lambda: configuration.config.encoded_strict_val_set)
)
