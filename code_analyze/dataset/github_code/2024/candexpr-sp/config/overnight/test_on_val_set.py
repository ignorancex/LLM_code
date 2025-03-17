
from itertools import chain

from dhnamlib.pylib.context import Environment, LazyEval
from dhnamlib.pylib.iteration import distinct_pairs

import configuration

from .test_general import config as _config_general


_config_specific = Environment(
    run_mode='test-on-val-set',
    train_domains=LazyEval(lambda: configuration.config.test_domains),
    # `train_domains` should be set,
    # because `domain.configuration._load_train_val_sets` requires `train_domains`.
    # The `_load_train_val_sets` function loads `encoded_val_set`.
)

config = Environment(distinct_pairs(chain(
    _config_general.items(),
    _config_specific.items()
)))
