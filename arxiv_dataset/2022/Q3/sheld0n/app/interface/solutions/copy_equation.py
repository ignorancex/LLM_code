#!/usr/bin/env python3

# Command line program
import argparse
# Name check
import re
# Directory operations
import shutil
import os
# Replace operations
import fileinput
# Custom Libraries
import sys
script_dir = os.path.dirname(os.path.realpath(__file__))
lib_directory = script_dir[:script_dir.find("interface") + len("interface")]
sys.path.append(lib_directory)
import libchoose

def parse():
    # parser
    parser = argparse.ArgumentParser(description='copy an equation')
    parser.add_argument('source', help='specify the name of source equation')
    parser.add_argument('name', help='specify the name of the copy')
    # parse
    return parser.parse_args()

def main(source, name):
    name_check = re.compile('[][@!#$%^&*()<>?/\\|}{~:+.\'"]')
    if(name_check.search(name) == None):
        # copy
        shutil.copytree(source, name, symlinks=True)
        # edit
        libchoose.edit_choice(name, [source], [name])
        libchoose.edit_add_equation(name)
    else:
        print("ERROR: name shouldn't contain special characters")

if __name__ == '__main__':
    # main
    args = parse()
    main(args.source, args.name)
