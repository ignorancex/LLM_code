import argparse
import math


def get_cmd_arg_dict():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-train-examples', dest='num_train_examples', type=int, help='the number of train examples to be used')

    args, unknown = parser.parse_known_args()

    return vars(args)
