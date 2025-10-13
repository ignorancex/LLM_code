#!/usr/bin/python
#Title			: compute_invariants_sets.py
#Usage			: python compute_invariants_sets.py -h
#Author			: pmorvalho
#Date			: April 27, 2021
#Description	        : Collects invariants from all students exercises into a pickle file
#Notes			: This can be used with similarity measures
#Script Version		: 1.0
#Python Version	: 3.8.5
# (C) Copyright 2021 Pedro Orvalho.
#==============================================================================

from sys import argv
import argparse
import re
import numpy as np
import pickle
import os
import gzip

def compute_program_invariants(input_file):
    #input: a file with a program's invariants
    #output: numpy array with the set of invariants ([[scope1, inv1, inv2]], scope2,...)
    lines = open(input_file, "r+").readlines()
    scope = False
    invariants = []
    for l in range(len(lines)):
        if re.match('.*:::ENTER$', lines[l]): # entering scope invariants
            scope = True
            scope_invariants=[]
            scope_name = lines[l].split("(")[0][2:]
            if re.match('_.*_.*', scope_name): # name of the current scope (injected functions e.g. _scope_depth_number)
                name_splt = scope_name.split("_")
                if args.scope_name:
                    scope_invariants.append('_{n}_'.format(n=name_splt[1]))
                if args.scope_depth:
                    scope_invariants.append("_depth_"+name_splt[2])
            elif scope_name != "main":
                if args.scope_name:
                    scope_invariants.append("function")

            continue
        elif re.match('=+$', lines[l]): # end of scope invariants =======
            if scope and len(scope_invariants) != 0:
                invariants.append(scope_invariants)
            scope = False
        if scope:
        # if scope and "one of" not in lines[l]:    
            scope_invariants.append(lines[l].strip("\n"))

    return np.array(invariants, dtype=object)

def compute_exercise_invariants(dir):
    # input: a exercise directory
    # output: a pickle with numpy of size 2 of equal size
    #         first entry is the name of each correct_submission
    #         second entry are the set of invariants of each submission
    ex_names = []
    ex_invariants = []
    exercises = []
    for d in os.listdir(dir):
        inv_file = dir+"/"+d+"/invariants/invariants.txt"
        if not os.path.isfile(inv_file):
            continue
        ex_names.append(d)
        invs = compute_program_invariants(inv_file)
        if len(invs)!=0:
            ex_invariants.append(invs)
            exercises.append(d)

    return np.array([exercises, ex_invariants], dtype=object)

def save_invariants_list(array):
    # computes the pickle name according to the features added. Scope name, scope depth or nothing
    if args.output_file is None:
        pickle_name=args.dir.replace("/submissions/","")+'/invariants'
        if args.scope_name:
            pickle_name += "-scope_name"
        if args.scope_depth:
            pickle_name += "-scope_depth"
        pickle_name += '.pickle'
    else:
        pickle_name = args.output_file + '.pickle'

    with open(pickle_name, "wb") as file:
        pickle.dump(array, file)

    os.system("gzip -f "+pickle_name)

def parser():
    parser = argparse.ArgumentParser(prog='compute_invariants_sets.py', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-d', '--dir',
                        help="directory with submissions for a given exercise")
    parser.add_argument('-n','--scope_name', action="store_true", default=False, help="computes the invariants for each scope and saving the scope's name along with the invariants")
    parser.add_argument('-dp', '--scope_depth', action="store_true", default=False, help="computes the invariants for each scope and save its name and depth as well")
    parser.add_argument('-o', '--output_file', nargs='?', help='the name of the output file.')
    args = parser.parse_args(argv[1:])
    return args

if __name__ == '__main__':
    args = parser()
    invs = compute_exercise_invariants(args.dir)
    save_invariants_list(invs)
