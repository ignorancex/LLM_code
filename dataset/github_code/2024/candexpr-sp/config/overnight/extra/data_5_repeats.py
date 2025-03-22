
from dhnamlib.pylib.context import Environment
from dhnamlib.pylib.context import LazyEval

from domain.overnight.configuration import _load_val_set, _load_test_set


def _get_repeated_dataset(dataset, num_repeats=5):
    repeated_dataset = dataset * num_repeats
    return repeated_dataset


config = Environment(
    encoded_val_set=LazyEval(lambda: _get_repeated_dataset(_load_val_set())),
    encoded_test_set=LazyEval(lambda: _get_repeated_dataset(_load_test_set())),
)
