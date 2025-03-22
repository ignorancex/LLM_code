
import argparse

from dhnamlib.pylib.text import parse_bool


def parse_args():
    parser = argparse.ArgumentParser(description='Learning a semantic parser for weak supervision')
    parser.add_argument('--pretrained-model-path', dest='pretrained_model_path', help='a path to a pretrained model')
    parser.add_argument('--resuming', dest='resuming', type=parse_bool, default=False, help='whether to resume learning')
    args, unknown = parser.parse_known_args()

    return vars(args)
