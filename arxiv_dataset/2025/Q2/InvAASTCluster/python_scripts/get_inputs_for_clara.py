#!/usr/bin/python
#Title			: get_inputs_for_clara.py
#Usage			: python get_inputs_for_clara.py labX/exY
#Author			: pmorvalho
#Date			: September 07, 2021
#Description	        : Prints the input tests in list format for a given exercise of a given lab. Which is the format accepted by CLARA (PLDI 2018)
#Notes			:
#Python Version	: 3.8.5
# (C) Copyright 2021 Pedro Orvalho.
#==============================================================================

from sys import argv
import glob

def get_values(tests):
	values=[]
	for t in tests:
		lines = open(t, "r+").readlines()
		t_in = []
		if lines == []:
			t_in = [""]
		for l in lines:
			t_in += l.split(" ")
		for n in range(len(t_in)):
			# Since CLARA only accepts integer inputs and outputs, we are going to return "" if the input is not an int
			try:
				t_in[n] = float(t_in[n])
			except Exception as e:
				t_in[n] = r"\"{s}\"".format(s=t_in[n])
		values.append(t_in)
	return values

if __name__ == '__main__':
	dir = argv[1]
	tests = glob.glob('tests/{d}/*.in'.format(d=dir), recursive=True)
	inputs = get_values(tests)
	print(str(inputs).replace("'","").replace(" ",""), end = '')
	
