#!/usr/bin/python
#Title			: find_closest_program.py
#Usage			: python find_closest_program.py [-h] -b bag_of_words.pickle -p program_name -pb bag_of_words_with_provided_program.pickle
#Author			: pmorvalho
#Date			: August 23, 2021
#Description    	: Given a program finds the closest program/programs from a given set of programs based on some Bag of Words (bow)
#Notes			:
#Python Version	: 3.8.5
# (C) Copyright 2021 Pedro Orvalho.
#==============================================================================

import argparse
import numpy as np
import pandas as pd
import re
from scipy.spatial import distance
from sys import argv

def get_submissions_name(l):
    t = "-correct"
    if "incorrect" in l:
        t = "-incorrect"
    d = l.split("/")[-1].split(".")[0]
    e = l.split("/")[-2]
    lab = l.split("/")[-3]
    ds = l.split("/")[-4]
    return "{ds}-{lab}-{e}-{d}{ty}".format(ds=ds, lab=lab, e=e, d=d,  ty='' if ds == "C-Pack-IPAs" else t)
    sub_name = re.sub("injected_programs/", "", l)
    sub_name = re.sub("ASTs/", "", sub_name)
    sub_name = re.sub("daikon-invariants/", "", sub_name)
    sub_name = re.sub(".*?_submissions/", "", sub_name)
    sub_name = re.sub("\.c", "", sub_name)
    sub_name = re.sub("\.ast", "", sub_name)
    sub_name = re.sub("invariants.*?gz", "", sub_name)
    sub_name = re.sub("/", "-", sub_name)
    return sub_name

def load_bow_pickle(bow_pickle):
    bp = pd.read_pickle(bow_pickle)
    return bp

def get_programs_info(bow_pickle):
    # returns submissions, submissions_vectors
    return bow_pickle['submissions'], bow_pickle['vectors']

def get_program_vector(program, bow_pickle):
    # returns the vector from a given program name
    idx=-1
    for p in range(len(bow_pickle['submissions'])):
        if program in bow_pickle['submissions'][p]:
            idx=p
    if idx == -1:
        return None
    return bow_pickle['vectors'][idx]

def calculate_program_distances(input_program_vec, submissions_vectors):
    # input:  input_program (vector) and a list of submissions_vectors
    # output: a list with distances between the given input_program and each submission
    distances = []
    if type(input_program_vec)!=list:
        input_program_vec = input_program_vec.toarray().tolist()[0]
    for s in range(len(submissions_vectors)):
        if type(submissions_vectors[s])!=list:
            sub_vec = submissions_vectors[s].toarray().tolist()[0]
        else:
            sub_vec = submissions_vectors[s]
        # print(len(sub_vec))
        distances.append(distance.euclidean(input_program_vec,sub_vec))

    return distances

def get_closest_programs(input_program_vector, submissions_vectors, submissions, verbose=False):
    distances = calculate_program_distances(input_program_vector, submissions_vectors)
    programs = []
    if distances == []:
        return []
    if verbose:
        print("# Correct submissions: ", len(submissions_vectors))
        print("# Closest  submissions: ", distances.count(min(distances)))
        # print("Closest  submission: ", submissions[distances.index(min(distances))])
        print("Closest  submissions: ")
    minimum = min(distances)
    idx=distances.index(minimum)
    programs.append(submissions[idx])
    if verbose:
        print(submissions[idx])
    try:
        while True:
            idx=distances.index(minimum, idx+1, len(distances))
            programs.append(submissions[idx])
            if verbose:
                print(submissions[idx])
    except Exception as e:
        pass
    return programs

def parser():
    parser = argparse.ArgumentParser(prog='find_closest_program.py', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-b', '--bow', type=str, help='Path to Bag of Words model (pickle file) to be used.')
    parser.add_argument('-p', '--program', type=str, help='Path to input program.')
    parser.add_argument('-pb', '--program_bow', type=str, help='Path to Bag of Words model (pickle file) with the input program vector.')
    parser.add_argument('-v', '--verbose', action='store_true', default=False, help='Prints debugging information.')
	# parser.add_argument('-ng', '--ngrams', default=3, type=int, help='')
	# parser.add_argument('-o', '--output_dir', nargs='?', help='the name of the output file.')
    args = parser.parse_args(argv[1:])
    return args

if __name__ == '__main__':
    args = parser()
    program_id = get_submissions_name(args.program)
    prog_bow = load_bow_pickle(args.program_bow)
    # print(prog_bow)
    # print(prog_bow['vectors'])
    # print(prog_bow['submissions'])
    bow = load_bow_pickle(args.bow)
    submissions, submissions_vectors = get_programs_info(bow)
    if args.verbose:
        print("Provided submission: ", program_id)
    input_program_vector = get_program_vector(program_id, prog_bow)
    # print(input_program_vector)
    if input_program_vector == None: # is None
        print("No available information about this submission in the Bag of Words model.")
        # no invariants
        exit()
    if args.verbose:
        print(input_program_vector.shape)
    # print(submissions)
    # print(submissions_vectors)
    get_closest_programs(input_program_vector, submissions_vectors, submissions)
