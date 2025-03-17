
from itertools import chain

from dhnamlib.pylib.context import Environment, LazyEval
from dhnamlib.pylib.iteration import distinct_pairs

from .test_general import config as _config_general


_config_specific = Environment(
    run_mode='test-on-test-set',
)

config = Environment(distinct_pairs(chain(
    _config_general.items(),
    _config_specific.items()
)))
