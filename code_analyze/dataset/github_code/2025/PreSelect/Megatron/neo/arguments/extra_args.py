# Copyright (c) 2023, NEO CORPORATION. All rights reserved.

"""Megatron extra arguments."""

import argparse


def add_neo_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group(title='data shuffle argument group')

    group.add_argument('--enable-shuffle', action='store_true',
                          help='Enable shuffle of the data')

    return parser

