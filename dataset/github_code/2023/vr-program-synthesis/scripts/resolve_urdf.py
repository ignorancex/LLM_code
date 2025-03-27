#!/usr/bin/env python

from __future__ import print_function
import argparse

from vr_program_synthesis.utils import resolve_urdf


def main(args):
    urdf_str = resolve_urdf(args.input_urdf)
    with open(args.output_file, "w") as output_file:
        output_file.write(urdf_str)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("input_urdf", type=str)
    parser.add_argument("output_file", type=str)
    main(parser.parse_args())
