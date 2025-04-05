#!/usr/bin/env python3

from pathlib import Path
import ntpath
import re
import subprocess
import os

output_dir = "output/"
example_dir_prefix = "tests/"
folders = ("singly-linked-list", "sll-sorted", "binary-search-tree", "misc")


safeSymbol = "\\ding{51}"
unsafeSymbol = "\\ding{55}"

incoherentSymbol = "no"
coherentSymbol = "yes"

notAvailableSymbol = "---"


def parse_subprocess_output(res):
    t = re.findall("\d+\.\d+", res)
    v = re.findall("unsafe", res)
    inc = re.findall("incoherent", res)
    n = re.findall("(\d+) states", res)
    user_time_str = t[0]
    safety = notAvailableSymbol
    num_states = notAvailableSymbol
    if inc == []:
        incoherent = coherentSymbol
        if v == []:
            safety = safeSymbol
            num_states = n[0]
        else:
            safety = unsafeSymbol
    else:
        incoherent = incoherentSymbol
    return user_time_str, safety, num_states, incoherent


def info_string(basename, loc, time, safety, incoherent, num_states):
    s = basename + " & " + loc + " & " + incoherent + " & "\
        + safety + " & " + num_states + " & " + time + " \\\\"
    return s

def print_info_string(basename, loc, time, safety, incoherent, num_states):
    s = info_string(basename, loc, time, safety, incoherent, num_states)
    print(s)

for folder in folders:
    pathlist = Path(example_dir_prefix+folder).glob('**/*.unint')
    path_in_str_list = []
    for path in pathlist:
        # because path is object not string
        path_in_str = str(path)
        path_in_str_list.append(path_in_str)
    sorted_list = sorted(path_in_str_list)
        
    for path_in_str in sorted_list:
        res = subprocess.check_output(["scripts/time.sh", "./_build/default/main.exe", path_in_str, "false"], universal_newlines=True, stderr=subprocess.STDOUT)
        user_time_str, safety, num_states, incoherent = parse_subprocess_output(res)
        basename = ntpath.basename(path_in_str)
        basename_extensionless = ntpath.splitext(basename)[0]
        with open(path_in_str) as fh:
            loc = 0
            for line in fh:
                if line != "\n": # remove empty lines
                    loc += 1
        info = info_string(str(basename_extensionless), str(loc), str(user_time_str), str(safety), str(incoherent), str(num_states))
        print(info)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_file = output_dir+basename_extensionless+".out"
        with open(output_file, 'w+') as fh:
            fh.write(info)
            fh.write("\n\n----------- RAW OUTPUT -----------\n\n")
            fh.write(res)



    
