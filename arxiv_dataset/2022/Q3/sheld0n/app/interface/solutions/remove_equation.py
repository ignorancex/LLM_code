#!/usr/bin/env python3

# Command line program
import argparse
# Directory operations
import shutil
import os
# Replace operations
import fnmatch
import fileinput
# Custom Libraries
import sys
script_dir = os.path.dirname(os.path.realpath(__file__))
lib_directory = script_dir[:script_dir.find("interface") + len("interface")]
sys.path.append(lib_directory)
import libchoose

def parse():
    parser = argparse.ArgumentParser(description='remove equations')
    parser.add_argument('equations', nargs='+', help='specify the names of the equations to remove')
    return parser.parse_args()

def edit(name):
    upper_camel_name = libchoose.object_to_upper_camel_case([name])
    # solutions
    has_removed_first = False
    for line in fileinput.FileInput("parameters.h", inplace=True):
        if (line == '#include "param/solutions/{}/parameters.h"\n'.format(name)):
            pass
        elif (line == '\t\t_{Name}\n'.format(Name=upper_camel_name)):
            has_removed_first = True
            pass
        elif (line == '\t\t,_{Name}\n'.format(Name=upper_camel_name)):
            pass
        else:
            if has_removed_first:
                line = line.replace(',', '')
                has_removed_first = False
            print(line, end='')

def main(name):
    # edit
    edit(name)
    # remove dir
    shutil.rmtree(name)
    # shutil.rmtree("../../../init/solutions/" + name)

if __name__ == '__main__':
    # main
    args = parse()
    for obj in args.equations:
        main(os.path.normpath(obj))
