from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import argparse


def args_parse():
    args = args_read()
    args = args_setup()
    return args


def args_read():
    parser = argparse.ArgumentParser()
    add = parser.add_argument
    args = parser.parse_args()
    return args


def args_setup(args):
    """Perform any additional operation on command-line arguments"""
    return args


