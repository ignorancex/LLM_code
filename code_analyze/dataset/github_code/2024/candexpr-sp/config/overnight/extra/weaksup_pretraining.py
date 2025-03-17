import math

import configuration
from domain.overnight import configuration as overnight_configuration

from dhnamlib.pylib.context import Environment
from dhnamlib.pylib.context import LazyEval

from ...common.extra.train_set_portion import compute_num_epoch_repeats


def _make_config():
    return Environment(
        num_epoch_repeats=LazyEval(lambda: compute_num_epoch_repeats(
            ratio=len(configuration.config.encoded_weaksup_pretraining_set) / overnight_configuration.NUM_ALL_DOMAIN_TRAIN_EXAMPLES,
            epoch_repeat_strategy='linear'
        )),
        encoded_train_set=LazyEval(lambda: configuration.config.encoded_weaksup_pretraining_set),
    )


config = _make_config()
