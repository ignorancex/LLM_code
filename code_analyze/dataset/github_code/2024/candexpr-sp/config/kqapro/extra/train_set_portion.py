
from dhnamlib.pylib.context import Environment, LazyEval
from dhnamlib.pylib.mllib.dataproc import extract_portion

import configuration

from ...common.extra.train_set_portion import get_cmd_arg_dict


def _extract_dataset_portion(dataset, percent, expected_dataset_size=None):
    extracted_dataset = extract_portion(dataset, percent=percent)
    if expected_dataset_size is not None:
        assert len(extracted_dataset) == expected_dataset_size    
    return extracted_dataset


def make_config(shuffled_encoded_train_set_name, total_train_set_size=None):
    cmd_arg_dict = get_cmd_arg_dict(default_epoch_repeat_strategy='sqrt')

    if total_train_set_size is not None:
        num_used_train_examples = round(total_train_set_size * cmd_arg_dict['train_set_percent'] / 100)

    if cmd_arg_dict['train_set_percent'] == 100:
        encoded_train_set = LazyEval(lambda: configuration.config.__getattr__(shuffled_encoded_train_set_name))
    else:
        encoded_train_set = LazyEval(lambda: _extract_dataset_portion(
            configuration.config.__getattr__(shuffled_encoded_train_set_name),
            percent=cmd_arg_dict['train_set_percent'],
            expected_dataset_size=num_used_train_examples))

    return Environment(
        num_epoch_repeats=cmd_arg_dict['num_epoch_repeats'],
        num_used_train_examples=num_used_train_examples,
        encoded_train_set=encoded_train_set,
    )


KQAPRO_TRAIN_SET_SIZE = 94376


config = make_config(
    'shuffled_encoded_train_set',
    total_train_set_size=KQAPRO_TRAIN_SET_SIZE)
