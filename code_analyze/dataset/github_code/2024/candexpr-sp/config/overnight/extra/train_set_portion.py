
from dhnamlib.pylib.context import Environment, LazyEval

# import configuration

from ...common.extra.train_set_portion import get_cmd_arg_dict


def make_config():
    cmd_arg_dict = get_cmd_arg_dict(default_epoch_repeat_strategy='linear')

    return Environment(
        # num_epoch_repeats=cmd_arg_dict['num_epoch_repeats'],
        train_set_portion_percent=cmd_arg_dict['train_set_percent'],
    )


config = make_config()
