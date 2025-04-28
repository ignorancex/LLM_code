#!/usr/bin/env python3

import argparse
import numpy as np
import shlex
import subprocess
import glob
import fileinput
import re

def parse():
    parser = argparse.ArgumentParser(description='rename an equation')
    parser.add_argument('source', help='specify the name of the equation')
    parser.add_argument('name', help='specify the new name of the equation')
    return parser.parse_args()

def compute(source, name):
    cmd = "./copy_equation {source} {name}".format(source=source, name=name)
    print("INFO: running " + cmd)
    subprocess.call(shlex.split(cmd))
    cmd = "./remove_equation {source}".format(source=source)
    print("INFO: running " + cmd)
    subprocess.call(shlex.split(cmd))

def main(source, name):
    compute(source, name)

if __name__ == '__main__':
    # parse arguments
    args = parse()
    # call
    main(args.source, args.name)
