from itertools import chain

from dhnamlib.pylib.context import Environment
from dhnamlib.pylib.context import LazyEval
from dhnamlib.pylib.iteration import distinct_pairs
from dhnamlib.pylib.structure import AttrDict

import configuration
from domain.overnight import configuration as overnight_configuration

from .train_general import config as _config_general
from ..common.multiple_decoding_strategies import get_decoding_strategy_configs


_config_specific = Environment(
    run_mode='train-for-multiple-decoding-strategies',
    decoding_strategy_configs=get_decoding_strategy_configs(overnight_configuration._make_grammar),
    num_epoch_repeats=LazyEval(lambda: (overnight_configuration.NUM_ALL_DOMAIN_TRAIN_EXAMPLES / len(configuration.config.encoded_train_set))),
)

config = Environment(chain(
    distinct_pairs(chain(
        _config_general.items(),
        _config_specific.items()))
))
