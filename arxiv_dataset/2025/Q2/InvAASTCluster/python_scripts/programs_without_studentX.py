#!/usr/bin/python
#Title			: programs_without_studentX.py
#Usage			: python programs_without_studentX.py -h
#Author			: pmorvalho
#Date			: October 15, 2021
#Description    	: Returns a list of possible programs without a certain student (specified by their ID)
#Notes			:
#Python Version: 3.8.5
# (C) Copyright 2021 Pedro Orvalho.
#==============================================================================

import argparse
import glob
from sys import argv

def parser():
    parser = argparse.ArgumentParser(prog='programs_without_studentX.py', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('dir', help='directory with programs')
    parser.add_argument('student_submission', help='path to the student submission. The student we want to ignore.\n The program\'s name follows the structure path_name/exYY-studentID-DATE.c we only want the studentID')
    parser.add_argument('-o', '--output_dir', nargs='?', help='the name of the output dir.')
    # parser.add_argument('-v', '--verbose', action='store_true', default=False, help='Prints debugging information.')
    args = parser.parse_args(argv[1:])
    return args

if __name__ == '__main__':
    args = parser()
    output_list = []
    submissions = glob.glob(args.dir+'/*.c', recursive=True)
    stu_id = args.student_submission.split("/")[-1]
    if "-" in stu_id:
        stu_id = stu_id.split("-")[1] # to remove from path name previous directories and number and date of the submissions, we only want the student id
    for s in submissions:
        if stu_id not in s:
            output_list.append(s)
    # output_list.sort()
    print(" ".join(output_list))
