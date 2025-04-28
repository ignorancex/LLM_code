#!/usr/bin/env python3

# command line program
import argparse
# directory operations
import os
import glob
# file edit
import fileinput
import fnmatch
# regular expression
import re

def parse():
    parser = argparse.ArgumentParser(description='Parameter library that contains functions to set the parameters of the simulation.')
    return parser.parse_args()

def find_replace(file_pattern, text, replacement):
    files = glob.glob(file_pattern, recursive=True)
    if files:
        for file in files:
            lineno = 1
            data_array = []
            for line in fileinput.FileInput(file, inplace=True):
                text_match = re.search(text, line)
                if text_match:
                    print(re.sub(text, replacement, line), end='')
                    data_array.append({
                        "text":text_match[0], 
                        "replacement":replacement,
                        "lineno":lineno
                    })
                else:
                    print(line, end='')
                lineno += 1
            if data_array:
                for data in data_array:
                    print("INFO: {file}:{lineno}: {text} ---> {replacement}".format(file=file, **data))
            else:
                print("WARNING: " + text + " not found in " + file)
    else:
        print("WARNING: " + file_pattern + " not found")

def set_parameter(dest, param, value):
    find_replace(dest + "/parameters.h", r" {param} = [^\;]*;".format(param=param), " {param} = {value};".format(param=param, value=str(value)))
    find_replace(dest + "/parameters.py", r" {param} = [^\\n]*\n".format(param=param), " {param} = {value}\n".format(param=param, value=str(value)))

if __name__ == '__main__':
    parse()
