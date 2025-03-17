
from itertools import chain

from dhnamlib.pylib.context import Environment
from dhnamlib.pylib.context import LazyEval
from dhnamlib.pylib.iteration import distinct_pairs
from dhnamlib.pylib.structure import AttrDict
# from dhnamlib.pylib.decoration import deprecated

from domain.kqapro import configuration as kqapro_configuration
# import configuration

from .train_general import config as _config_general
from ..common.multiple_decoding_strategies import get_decoding_strategy_configs


_config_specific = Environment(
    run_mode='train-for-multiple-decoding-strategies',
    decoding_strategy_configs=get_decoding_strategy_configs(kqapro_configuration._make_grammar)
)

config = Environment(chain(
    distinct_pairs(chain(
        _config_general.items(),
        _config_specific.items()))
))
