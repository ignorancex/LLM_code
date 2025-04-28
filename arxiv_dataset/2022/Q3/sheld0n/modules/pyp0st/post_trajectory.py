#!/usr/bin/env python3

# command line program
import argparse
# deepcopy
import copy
# numpy
import numpy as np
# internal modules
import libpost

def parse():
    parser = argparse.ArgumentParser(description='Extract the trajectory of an object.')
    parser.add_argument('name', help='specify the name of the object')
    return parser.parse_args()

def main(name):
    # get positions
    trajectory = {name:libpost.get_objects_properties(["pos_0", "pos_1", "pos_2"])[name]}
    libpost.savet(libpost.get_time(), trajectory, "trajectory")

if __name__ == '__main__':
    args = parse()
    main(args.name)
