from itertools import chain
import configuration

from dhnamlib.pylib.context import Environment
from dhnamlib.pylib.context import LazyEval

from .train_set_portion import make_config, KQAPRO_TRAIN_SET_SIZE

config = Environment(chain(
    make_config('shuffled_encoded_strict_train_set', KQAPRO_TRAIN_SET_SIZE).items(),
    Environment(
        inferencing_subtypes=False,
        encoded_val_set=LazyEval(lambda: configuration.config.encoded_strict_val_set)).items(),
))
