#!/usr/bin/env python3

import argparse
import numpy as np
import shlex
import subprocess
import glob
import fileinput
import re
import os

def parse():
    parser = argparse.ArgumentParser(description='Copies an equation into several others by varying their parameters.')
    parser.add_argument('sources', nargs='+', help='specify the name of the source equations')
    parser.add_argument('-p', '--prop', help='specify the property', required=True)
    parser.add_argument('-v', '--values', type=float, nargs='+', help='specify the values', required=True)
    parser.add_argument('-f', '--factor', help='specify a factor', default=None)
    return parser.parse_args()

def get_property_from_dir_name(name, prop):
    all_properties = name.split("__")
    for prop_name in all_properties:
        if prop_name.startswith(prop):
            return float(prop_name[len(prop)+1:].replace("o", ".").replace("m", "-"))
    return np.nan

def set_property_from_dir_name(name, prop, value, fwidth, fprecision):
    is_property_set = False
    all_properties = name.split("__")
    for i in range(len(all_properties)):
        if all_properties[i].startswith(prop):
            all_properties[i] = prop + "_" + "{value:0>{fwidth}.{fprecision}f}".format(value=value, fwidth=fwidth, fprecision=fprecision).replace(".", "o").replace("-", "m")
            is_property_set = True
    if not is_property_set:
        all_properties.append(prop + "_" + "{value:0>{fwidth}.{fprecision}f}".format(value=value, fwidth=fwidth, fprecision=fprecision).replace(".", "o").replace("-", "m"))
    name = ""
    for prop_name in all_properties:
        name += prop_name + "__"
    return name[:-2]

def file_replace(file_name, text, replacement):
    lineno = 1
    data_array = []
    if os.path.exists(file_name):
        for line in fileinput.FileInput(file_name, inplace=True):
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
            print("INFO: {file}:{lineno}: {text} ---> {replacement}".format(file=file_name, **data))

def set_parameter(dest, prop, value, factor):
    if factor:
        file_replace(dest + "/parameters.h", r"{prop} = [^\;]*;".format(prop=prop), "{prop} = {value} * {factor};".format(prop=prop, value=str(value), factor=factor))
        file_replace(dest + "/parameters.py", r"{prop} = .*$".format(prop=prop), "{prop} = {value} * {factor}".format(prop=prop, value=str(value), factor=factor))
    else:
        file_replace(dest + "/parameters.h", r"{prop} = [^\;]*;".format(prop=prop), "{prop} = {value};".format(prop=prop, value=str(value)))
        file_replace(dest + "/parameters.py", r"{prop} = .*$".format(prop=prop), "{prop} = {value}".format(prop=prop, value=str(value)))

def compute(sources, prop, values, factor):
    finteger = 0
    fprecision = 0
    for value in values:
        str_value = str(value)
        str_value_array = str_value.split(".")
        if len(str_value_array) == 2:
            if len(str_value_array[0]) > finteger:
                finteger = len(str_value_array[0])
            if len(str_value_array[1]) > fprecision:
                fprecision = len(str_value_array[1])
    fwidth = finteger + fprecision + 1
    for source in sources:
        for value in values:
            dest = set_property_from_dir_name(source, prop, value, fwidth, fprecision)
            # execute command
            cmd = "./copy_equation {source} {dest}".format(source=source, dest=dest)
            print("INFO: running " + cmd)
            subprocess.call(shlex.split(cmd))
            # set parameter
            set_parameter(dest, prop, value, factor)

def main(source, prop, values, factor):
    compute(source, prop, values, factor)

if __name__ == '__main__':
    # parse arguments
    args = parse()
    # call
    main(args.sources, args.prop, args.values, args.factor)
