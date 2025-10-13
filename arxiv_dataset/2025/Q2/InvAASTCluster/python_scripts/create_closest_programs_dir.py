#!/usr/bin/python
#Title			: create_closest_programs_dir.py
#Usage			: python create_closest_programs_dir.py -h
#Author			: pmorvalho
#Date			: September 09, 2021
#Description    	: Creates for each incorrect program two new directories with the first closest program using some Bag of Words, and another with all programs at the same distance if more than one
#Notes			:
#Python Version	: 3.8.5
# (C) Copyright 2021 Pedro Orvalho.
#==============================================================================


import argparse
from sys import argv
from find_closest_program import *
import glob
import os

def recreate_submissions_name(s):
	if "C-Pack-IPAs" in s:
		s = re.sub("-lab", "/lab", s)
		s = re.sub("-ex", "/ex", s)
		s = re.sub("/ex[0-9]*-[altp]{2}[0-9]*-[0-9]*/", "/", s)
	else:
		s = s.replace("-correct", "")
		s = re.sub("-", "/", s)
	s = "correct_programs_clara_supports/{s}.c".format(s=s)
	return s

def parser():
	parser = argparse.ArgumentParser(prog='create_closest_programs_dir.py', formatter_class=argparse.RawTextHelpFormatter)
	parser.add_argument('-cd', '--correct_dir', nargs='?', type=str, help='Path to directory with correct submissions. E.g. correct_submissions/lab02/ex01')
	parser.add_argument('-ip', '--incorrect_program', type=str, help='Name of incorrect submission.')
	parser.add_argument('-cb', '--correct_bow', type=str, help='Path to Bag of Words model (pickle file) with correct submissions.')
	parser.add_argument('-o', '--output_dir', type=str, help='Name of the output directory.')
	parser.add_argument('-ib', '--incorrect_bow', type=str, help='Path to Bag of Words model (pickle file) with incorrect submissions.')
	parser.add_argument('-v', '--verbose', action='store_true', default=False, help='Prints debugging information.')
	# parser.add_argument('-ng', '--ngrams', default=3, type=int, help='')
	# parser.add_argument('-o', '--output_dir', nargs='?', help='the name of the output directory.')
	args = parser.parse_args(argv[1:])
	return args

if __name__ == '__main__':
	args = parser()

	# incorrect_programs = glob.glob('{d}/*.c'.format(d=args.incorrect_dir), recursive=True)
	bow = load_bow_pickle(args.correct_bow)
	# print(bow)
	bow_name = args.correct_bow.split("/")[-1].replace(".pickle.gz", "")
	submissions, submissions_vectors = get_programs_info(bow)
	# for ip in incorrect_programs:
	program_id = get_submissions_name(args.incorrect_program)
	# print(program_id)
	incorrect_bow = load_bow_pickle(args.incorrect_bow)
	# print(incorrect_bow)
	sub = args.incorrect_program.split("/")[-1]
	lab = args.incorrect_program.replace("incorrect_submissions/","").replace(sub,"")
	if args.correct_dir != None:
		clusters_representatives = glob.glob('{d}/*.c'.format(d=args.correct_dir), recursive=True)
		# we need to remote the path name and the cluster name to get the submission name representated in the Bag of Words
		if "C-Pack-IPAs" in lab:
			# clusters_representatives = [get_submissions_name(c) for c in clusters_representatives]
			clusters_representatives = [lab.replace("/","-")+"-".join(c.split("/")[-1].split("-")[1:]).replace(".c", "") for c in clusters_representatives]
		else:
			clusters_representatives = [lab.replace("/","-")+"-".join(c.split("/")[-1].split("-")[1:]).replace(".c", "")+"-correct" for c in clusters_representatives]
		clusters_representatives_vectors = []
		for c in clusters_representatives:
			clusters_representatives_vectors.append(submissions_vectors[submissions.index(c)])
		submissions = clusters_representatives
		submissions_vectors = clusters_representatives_vectors
	else:
		# we need to remove the same student's correct submissions when using all correct submissions
		stu_id = args.incorrect_program.split("/")[-1] # to remove from path name previous directories and number and date of the submissions, we only want the student id
		if "-" in stu_id:
			stu_id = stu_id.split("-")[1]
		stu_id = stu_id.replace(".c", "")
		for s in range(len(submissions)-1,-1,-1):
			if stu_id in submissions[s]:
				submissions.pop(s)
				submissions_vectors.pop(s)
	sub = sub.replace(".c", "")
	dir_name = args.output_dir
	os.system("mkdir -p {dn}/first_program/{lab}/{sub} {dn}/several_programs/{lab}/{sub}".format(dn=dir_name,lab=lab,sub=sub))

	input_program_vector = get_program_vector(program_id, incorrect_bow)
	# print(input_program_vector)
	if input_program_vector == None: # is None
		print("No available information about this submission in the Bag of Words model.")
		exit()
		# no invariants
	# print(submissions)
	# print(submissions_vectors)
	programs = get_closest_programs(input_program_vector, submissions_vectors, submissions, verbose=args.verbose)
	# print(programs)
	if programs == []:
		print("No programs found")
		exit()
	os.system("cp {p} {dn}/first_program/{lab}/{sub}/.".format(p=recreate_submissions_name(programs[0]),dn=dir_name,lab=lab,sub=sub))
	for p in programs:
		os.system("cp {p} {dn}/several_programs/{lab}/{sub}/.".format(p=recreate_submissions_name(p),dn=dir_name,lab=lab,sub=sub))
