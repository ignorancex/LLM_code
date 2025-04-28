#!/usr/bin/env python3

# command line program
import argparse
# util
import math
# lib param
import libset


def parse():
    parser = argparse.ArgumentParser(description='Set a simulation parameter.')
    parser.add_argument('file', type=str, help='parameter file')
    parser.add_argument('parameter', type=str, help='parameter name')
    parser.add_argument('value', type=str, help='new value of that parameter')
    return parser.parse_args()

def main():
    args = parse()
    ## set, the path to directories must be given following the glob syntax.
    libset.set_parameter(args.file, args.parameter, args.value)

if __name__ == '__main__':
    main()
