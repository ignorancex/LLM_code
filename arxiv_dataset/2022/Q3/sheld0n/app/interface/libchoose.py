#!/usr/bin/env python3

# Command line program
import argparse
# Name check
import re
# Directory operations
import os
import shutil
import glob
# File edit
import fileinput
import fnmatch

# getters
def get_selected():
    for name in [os.path.basename(os.path.dirname(path)) for path in glob.glob("./*/")]:
        if not name.startswith("_"):
            return name

def get_object():
    re_include = re.compile(r'^#include "param/solutions/.*/(parameters.h|choice.h)"')
    with open('choice.h', 'r') as reader:
        line = reader.readline()
        while line and re_include.match(line) == None:
            line = reader.readline()
    return line.split("/")[4:-2]

def get_object_type():
    re_include = re.compile(r'^#include "param/solutions/.*/(parameters.h|choice.h)"')
    with open('choice.h', 'r') as reader:
        line = reader.readline()
        while line and re_include.match(line) == None:
            line = reader.readline()
    return line.split("/")[3]

def get_flow():
    re_include = re.compile(r'^#include "param/flow/.*/(parameters.h|choice.h)"')
    with open('choice.h', 'r') as reader:
        line = reader.readline()
        while line and re_include.match(line) == None:
            line = reader.readline()
    return line.split("/")[2:-2]

def get_init():
    re_include = re.compile(r'^#include "param/init/solutions/.*/(parameters.h|choice.h)"')
    with open('choice.h', 'r') as reader:
        line = reader.readline()
        while line and re_include.match(line) == None:
            line = reader.readline()
    return line.split("/")[3:-2]

def get_post():
    re_include = re.compile(r'^#include "param/post/solutions/.*/(parameters.h|choice.h)"')
    with open('choice.h', 'r') as reader:
        line = reader.readline()
        while line and re_include.match(line) == None:
            line = reader.readline()
    return line.split("/")[3:-2]

def get_abs_choices_dir(choices_dir):
    script_dir = os.path.dirname(os.path.realpath(__file__))
    return script_dir[:script_dir.find("interface")-1] + "/default/param/" + choices_dir

def get_choices_dir(choices_dir, choices_exceptions = []):
    abs_choices_dir = get_abs_choices_dir(choices_dir)
    _choices = [os.path.basename(os.path.dirname(path)) for path in glob.glob("{abs_choices_dir}/*/".format(abs_choices_dir=abs_choices_dir))]
    # build choices dict
    choices = {}
    for choice in _choices:
        if choice.startswith('__'):
            choices[choice[2:]] = choice
        elif not choice.startswith('_'):
            choices[choice] = choice
    # remove exceptions
    for exception in choices_exceptions:
        choices.pop(exception, None)
    choices.pop(get_selected(), None)
    # reurn
    return choices


def get_default_object(abs_choices_dir):
    re_include = re.compile(r'^#include "param/solutions/.*/(parameters.h|choice.h)"')
    with open(abs_choices_dir + '/choice.h', 'r') as reader:
        line = reader.readline()
        while line and re_include.match(line) == None:
            line = reader.readline()
    return line.split("/")[4:-2]

def get_default_object_type(abs_choices_dir):
    re_include = re.compile(r'^#include "param/solutions/.*/(parameters.h|choice.h)"')
    with open(abs_choices_dir + '/choice.h', 'r') as reader:
        line = reader.readline()
        while line and re_include.match(line) == None:
            line = reader.readline()
    return line.split("/")[3]

def get_default_flow(abs_choices_dir):
    re_include = re.compile(r'^#include "param/flow/.*/(parameters.h|choice.h)"')
    with open(abs_choices_dir + '/choice.h', 'r') as reader:
        line = reader.readline()
        while line and re_include.match(line) == None:
            line = reader.readline()
    return line.split("/")[2:-2]

def get_default_init(abs_choices_dir):
    re_include = re.compile(r'^#include "param/init/solutions/.*/(parameters.h|choice.h)"')
    with open(abs_choices_dir + '/choice.h', 'r') as reader:
        line = reader.readline()
        while line and re_include.match(line) == None:
            line = reader.readline()
    return line.split("/")[3:-2]

def get_default_post(abs_choices_dir):
    re_include = re.compile(r'^#include "param/post/solutions/.*/(parameters.h|choice.h)"')
    with open(abs_choices_dir + '/choice.h', 'r') as reader:
        line = reader.readline()
        while line and re_include.match(line) == None:
            line = reader.readline()
    return line.split("/")[3:-2]

# edit

def str_to_upper_camel_case(s):
    return s.title().replace("_", "")

def object_to_upper_snake_case(obj):
    s = ""
    for o in obj:
        if o.startswith("__"):
            o = o[2:]
        elif o.startswith("_"):
            o = o[1:]
        s += o.upper() + "_"
    return s[:-1]

def object_to_upper_camel_case(obj):
    s = ""
    for o in obj:
        if o.startswith("__"):
            o = o[2:]
        elif o.startswith("_"):
            o = o[1:]
        s += o.title().replace("_", "")
    return s

def object_to_path(obj):
    s = ""
    for o in obj:
        s += o + "/"
    return s[:-1]

# def edit_header(selected, choice):
#     # edit choice.h
#     re_include = re.compile(r'^#include "core/solutions/object/.*/core.h"')
#     with open('choice.h', 'r') as reader:
#         line = reader.readline()
#         while line and re_include.match(line) == None:
#             line = reader.readline()
#     upper_camel_obj = object_to_upper_camel_case(line.split("/")[4:-2])
#     if not upper_camel_obj:
#         re_include = re.compile(r'^#include "param/solutions/.*/(parameters.h|choice.h)"')
#         with open('choice.h', 'r') as reader:
#             line = reader.readline()
#             while line and re_include.match(line) == None:
#                 line = reader.readline()
#         upper_camel_obj = object_to_upper_camel_case(line.split("/")[4:-2])
#     # replace
#     for line in fileinput.FileInput("choice.h", inplace=True):
#         line = line.replace("/" + selected + "/", "/" + choice + "/")
#         line = line.replace(upper_camel_obj + str_to_upper_camel_case(selected), upper_camel_obj + str_to_upper_camel_case(choice))
#         print(line, end='')

## source https://stackoverflow.com/questions/4205854/python-way-to-recursively-find-and-replace-string-in-text-files
def find_replace(folder, file_pattern, text, replacement, condition = lambda line : True):
    for dirpath, dirs, files in os.walk(folder, topdown=True):
        files = [os.path.join(dirpath, filename) for filename in fnmatch.filter(files, file_pattern)]
        if files:
            for line in fileinput.FileInput(files, inplace=True):
                if condition(line):
                    print(re.sub(text, replacement, line), end='')
                else:
                    print(line, end='')

def edit_add_equation(name):
    upper_camel_name = object_to_upper_camel_case([name])
    ## add equation to param/solutions/parameters.h
    previous_line = ''
    for line in fileinput.FileInput("parameters.h", inplace=True):
        if line == '// FLAG: INCLUDE EQUATION END\n':
            print('#include "param/solutions/{}/parameters.h"\n'.format(name), end='')
        if line == '\t\t// FLAG: DECLARE STATIC EQUATION END\n':
            if previous_line == '\t\t// FLAG: DECLARE STATIC EQUATION BEGIN\n':
                 print('\t\t_{Name}\n'.format(Name=upper_camel_name), end='')
            else:
                print('\t\t,_{Name}\n'.format(Name=upper_camel_name), end='')
        previous_line = line
        print(line, end='')

def edit_choice(choice, default_obj, obj, size = 1):
    obj = obj[0:len(obj) - len(default_obj) + size]
    default_obj = default_obj[0:size]
    find_replace(choice, "*.h", '"' + "".join(default_obj), '"' + "".join(obj))
    find_replace(choice, "*.h", "_" + object_to_upper_snake_case(default_obj) + "_", "_" + object_to_upper_snake_case(obj) + "_")
    find_replace(choice, "*.h", "/" + object_to_path(default_obj) + "/", "/" + object_to_path(obj) + "/", lambda line : line.startswith('#include "param'))
    find_replace(choice, "*.h", "_" + object_to_upper_camel_case(default_obj), "_" + object_to_upper_camel_case(obj))

def create_sym_links(choice, choice_alt):
    script_dir = os.path.dirname(os.path.realpath(__file__))
    interface_dir = script_dir[:script_dir.find("interface")-1] + "/interface/" + choices_dir + "/" + choice_alt
    if os.path.exists(interface_dir + "/choose.py"):
        os.symlink(os.path.relpath(interface_dir + "/choose.py", choice), choice + "/choose")
    for path in glob.glob(interface_dir + "/*/"):
        subdir = os.path.basename(os.path.dirname(path))
        if not subdir.startswith("__"):
            #if os.path.exists(path + "choose.py"):
            #    os.symlink(path + "choose.py", choice + "/" + subdir + "/choose")
            create_sym_links(choice + "/" + subdir, choice_alt + "/" + subdir)










# choose file

def get_choices_file(choices_dir, choices_exceptions = []):
    abs_choices_dir = get_abs_choices_dir(choices_dir)
    _choices = [os.path.basename(path) for path in glob.glob("{abs_choices_dir}/__*.h".format(abs_choices_dir=abs_choices_dir))]
    # build choices dict
    choices = {}
    for choice in _choices:
        if choice.startswith('__'):
            choices[choice[2:].split(".")[0]] = choice
        elif not choice.startswith('_'):
            choices[choice.split(".")[0]] = choice
    # remove exceptions
    for exception in choices_exceptions:
        choices.pop(exception, None)
    # reurn
    return choices

def parse_file(choices_dir, choices_exceptions):
    # get choices
    choices = list(get_choices_file(choices_dir, choices_exceptions).keys())
    choices.sort()
    # parser
    parser = argparse.ArgumentParser(description='Script to choose parameters among default ones.')
    parser.add_argument('choice', choices=choices, help='specify your choice')
    # parse
    return parser.parse_args()

def choose_file(choices_dir, choices_exceptions):
    args = parse_file(choices_dir, choices_exceptions)
    choices = get_choices_file(choices_dir, choices_exceptions)
    shutil.copyfile(get_abs_choices_dir(choices_dir) + "/" + choices[args.choice], "parameters.h")
    if args.choice.endswith("_py"):
        shutil.copyfile(get_abs_choices_dir(choices_dir) + "/" + choices[args.choice].replace(".h", ".py"), "parameters.py")
    else:
        if os.path.exists("parameters.py"):
            os.remove("parameters.py")

def edit_file(choice, default_obj, obj, size = 1):
    obj = obj[0:len(obj) - len(default_obj) + size]
    default_obj = default_obj[0:size]
    find_replace(".", "parameters.h", '"' + "".join(default_obj), '"' + "".join(obj))
    find_replace(".", "parameters.h", "_" + object_to_upper_snake_case(default_obj) + "_", "_" + object_to_upper_snake_case(obj) + "_")
    find_replace(".", "parameters.h", "/" + object_to_path(default_obj) + "/", "/" + object_to_path(obj) + "/", lambda line : line.startswith('#include "param'))
    find_replace(".", "parameters.h", "_" + object_to_upper_camel_case(default_obj), "_" + object_to_upper_camel_case(obj))
    find_replace(".", "parameters.py", '"' + "".join(default_obj), '"' + "".join(obj))
    find_replace(".", "parameters.py", "_" + object_to_upper_snake_case(default_obj) + "_", "_" + object_to_upper_snake_case(obj) + "_")
    find_replace(".", "parameters.py", "/" + object_to_path(default_obj) + "/", "/" + object_to_path(obj) + "/", lambda line : line.startswith('#include "param'))
    find_replace(".", "parameters.py", "_" + object_to_upper_camel_case(default_obj), "_" + object_to_upper_camel_case(obj))
