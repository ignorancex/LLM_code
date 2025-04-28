# read in a list of MIST isochrones, record parameters of interest

import sys, glob
import numpy as np

# input files
infiles = glob.glob('*/isochrones/*.iso')
fds = np.array([0, 1, 2, 3, 8, 9, 13, 15, 18, 22, 23, 25]) # fields we need from input files
fdnames = np.array(['EEP', 'log10_isochrone_age_yr', 'initial_mass',\
			'star_mass', 'log_L', 'log_L_div_Ledd', 'log_Teff', 'log_R', 'surf_avg_omega',\
			'surf_r_equatorial_div_r', 'surf_r_polar_div_r', 'surf_avg_omega_div_omega_crit'])

# output file
outfile = 'mist_isochrones.txt'
f_out = open(outfile, 'w') # open the output file for writing

for f in infiles:
	f_in = open(f, 'r') # open an input file for reading
	print(f)
	lastline = ''
	for line in f_in:
		if ('Yinit' in lastline):
			# then last line should have been
			#  Yinit        Zinit   [Fe/H]   [a/Fe]  v/vcrit
			# and this line should look like
			# 0.2640  1.01135E-02    -0.15     0.00     0.00
			data = line.split()
			f_out.write('# Zinit [Fe/H] otilde\n{} {} {}\n'.format(data[2], data[3], data[5]))
		elif ('# EEP' in line):
			# check that the fields are what we expect
			fieldnames = np.take(line.split(), fds + 1)
			if np.array_equal( fieldnames, fdnames ):
				f_out.write('# ')
				print(*fieldnames, sep=' ', end='\n', file=f_out)
			else:
				print(fieldnames)
		elif line[0] != '#' and line.strip():
			fields = np.take(line.split(), fds)
			print(*fields, sep=' ', end='\n', file=f_out)
		lastline = line
