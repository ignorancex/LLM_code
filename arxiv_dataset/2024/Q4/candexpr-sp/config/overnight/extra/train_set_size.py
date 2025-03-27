
from dhnamlib.pylib.context import Environment, LazyEval

# import configuration
from domain.overnight import configuration as overnight_configuration

from ...common.extra.train_set_size import get_cmd_arg_dict
from ...common.extra.train_set_portion import compute_num_epoch_repeats


def make_config():
    cmd_arg_dict = get_cmd_arg_dict()
    num_train_examples = cmd_arg_dict['num_train_examples']

    return Environment(
        num_epoch_repeats=compute_num_epoch_repeats(
            ratio=num_train_examples / overnight_configuration.NUM_ALL_DOMAIN_TRAIN_EXAMPLES,
            epoch_repeat_strategy='linear'),
        num_used_train_examples=num_train_examples,
        train_set_portion_percent=None,
    )


config = make_config()
