
import math

import configuration

from dhnamlib.pylib.context import Environment
from dhnamlib.pylib.context import LazyEval

from ...common.extra.train_set_portion import compute_num_epoch_repeats


KQAPRO_TRAIN_SET_SIZE = 94376


def _make_config():
    sub_dataset_lazy_obj = LazyEval(lambda: configuration.config.encoded_weaksup_pretraining_set)
    return Environment(
        num_epoch_repeats=LazyEval(lambda: compute_num_epoch_repeats(
            ratio=len(sub_dataset_lazy_obj.get()) / KQAPRO_TRAIN_SET_SIZE,
            epoch_repeat_strategy='sqrt'
        )),
        num_used_train_examples=LazyEval(lambda: len(sub_dataset_lazy_obj.get())),
        encoded_train_set=LazyEval(sub_dataset_lazy_obj.get),
    )


config = _make_config()
