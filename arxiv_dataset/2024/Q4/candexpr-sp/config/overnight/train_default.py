
from itertools import chain

from dhnamlib.pylib.context import Environment, LazyEval
from dhnamlib.pylib.iteration import distinct_pairs

import configuration
from domain.overnight import configuration as overnight_configuration
from .train_general import config as _config_general


_config_specific = Environment(
    run_mode='train-default',
    num_epoch_repeats=LazyEval(lambda: (overnight_configuration.NUM_ALL_DOMAIN_TRAIN_EXAMPLES / len(configuration.config.encoded_train_set))),
)

config = Environment(distinct_pairs(chain(
    _config_general.items(),
    _config_specific.items()
)))
