import argparse
import math


def get_cmd_arg_dict(default_epoch_repeat_strategy='sqrt'):
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-set-percent', dest='train_set_percent', type=float, help='train set percent to be used')
    parser.add_argument('--num-epoch-repeats', dest='num_epoch_repeats', type=float, help='the number of epoch repeats before validation')
    parser.add_argument('--epoch-repeat-strategy', dest='epoch_repeat_strategy', type=str, choices=['linear', 'sqrt'],
                        default=default_epoch_repeat_strategy,
                        help='the way to compute the number of epoch repeats')

    args, unknown = parser.parse_known_args()

    train_set_percent = args.train_set_percent
    assert 0 < train_set_percent <= 100

    if args.num_epoch_repeats is None:
        num_epoch_repeats = compute_num_epoch_repeats(
            percent=args.train_set_percent,
            epoch_repeat_strategy=args.epoch_repeat_strategy)
        epoch_repeat_strategy = args.epoch_repeat_strategy
    else:
        num_epoch_repeats = args.num_epoch_repeats
        epoch_repeat_strategy = 'none'

    cmd_arg_dict = dict(
        train_set_percent=train_set_percent,
        num_epoch_repeats=num_epoch_repeats,
        epoch_repeat_strategy=epoch_repeat_strategy,
    )
    return cmd_arg_dict


def compute_num_epoch_repeats(ratio=None, percent=None, *, epoch_repeat_strategy):
    if ratio is None:
        assert percent is not None
        _reverse_ratio = 100 / percent
    else:
        assert percent is None
        _reverse_ratio = 1 / ratio

    if epoch_repeat_strategy == 'linear':
        num_epoch_repeats = _reverse_ratio
    elif epoch_repeat_strategy == 'sqrt':
        num_epoch_repeats = math.sqrt(_reverse_ratio)
    else:
        raise Exception('Unknown epoch_repeat_strategy')
    return num_epoch_repeats
