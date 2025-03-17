
from itertools import chain

from dhnamlib.pylib.context import Environment
# from dhnamlib.pylib.context import LazyEval
from dhnamlib.pylib.iteration import distinct_pairs

from .test_general import config as _config_general


_config_specific = Environment(
    # specific config here
)

_config_overriding = Environment(
    # specific config here
    run_mode='test-on-test-set',
)

config = Environment(chain(
    distinct_pairs(chain(
        _config_general.items(),
        _config_specific.items())),
    _config_overriding.items()))
