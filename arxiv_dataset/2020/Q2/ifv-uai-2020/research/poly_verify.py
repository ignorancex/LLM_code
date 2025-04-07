# Author: Philips George John

import sys, os
sys.path.insert(0, '../dev-pkgs')
sys.path.insert(0, '.')

import time

import numpy as np
import random as rnd
import pandas as pd

from sklearn.externals import joblib
from sklearn.svm import SVC

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

import joblib
from joblib import Parallel, delayed

from ibv.model import InputVariable

from polynomial import Polynomial
from sos_verifier import SosVerifier
from itertools import combinations

NUM_JOBS = 24

def svm2poly(sv_weights, sv_x, bias, gamma, r, d):
	assert(sv_weights.shape[0] == sv_x.shape[0])
	n        = sv_x.shape[1]
	N        = Polynomial.num_monomials(n, d)
	P_coefs  = np.zeros(shape = (sv_weights.shape[0], N))
	for i in range(sv_weights.shape[0]):
		P = Polynomial.affine_power_polynomial(v = sv_x[i], gamma = gamma, r = r, d = d)
		P_coefs[i] = np.array(P.coef_)[:-1]
	# End for
	C = np.dot(P_coefs.T, sv_weights)
	C[0] += bias
	C = C.tolist()

	return Polynomial.from_coefs(n, d, C)
# End svm2poly

def write_sage_script(svm_P, input_vars, fname = 'test.sage'):
	with open(fname, 'w') as f:
		n = len(input_vars)
		prot = min(i for i in range(n) if input_vars[i].is_protected == True)

		print('\nimport sys\n', file = f)
		print('from CvxAlgGeo import *\n', file = f)
		print('Rng = PolynomialRing(QQ, \'x\', %d)' % n, file = f)
		print('x = Rng.gens();', file = f)
		print('Prog  = []', file = f)
		print('Cons = []', file = f)
		print('conf = {\'Solver\':\'csdp\', \'detail\':True, \'AutoMatrix\':True}', file = f)
		print('\nprint \'Generating SoS program...\\n\'\n', file = f)
		print('\nP =', svm_P.sage_declaration(), file = f)
		P0 = svm_P.copy()
		P1 = svm_P.copy()
		P0.substitute({prot:1})
		print('\nP0 =', P0.sage_declaration(), file = f)
		P1.substitute({prot:2})
		print('\nP1 =', P1.sage_declaration(), file = f)
		print('\nF = P0 * P1', file = f)
		print('Prog.append(F)', file = f)
		print('\n# Range constraints for non-protected attributes', file = f)
		for i in range(n):
			if i != prot:
				print('Cons.append(x[%d] - %.3f)' % (i, input_vars[i].lower_bound), file = f)
				print('Cons.append(-x[%d] + %.3f)' % (i, input_vars[i].upper_bound), file = f)
		# End for
		print('\nProg.append(Cons)', file = f)
		print('\n#print Prog', file = f)
		print('\nprint \'SoS program generated...\\n\'\n', file = f)

	#	print('\nprint \'Initializing geometric.GPTools instance...\'', file = f)
	#	print('sys.stdout.flush()', file = f)
	#	print('A = geometric.GPTools(Prog, Rng, Settings = conf)', file = f)
	#	print('print \'Minimizing Geometric program...\'', file = f)
	#	print('sys.stdout.flush()', file = f)
	#	print('A.minimize()', file = f)
	#	print('print A.Info', file = f)

		print('\nprint \'Initializing semidefinite.SosTools instance...\'', file = f)
		print('sys.stdout.flush()', file = f)
		print('B = semidefinite.SosTools(Prog, Rng, Settings = conf)', file = f)
		print('\nprint \'Solving degree %d SoS...\' % B.Relaxation', file = f)
		print('print \'Initializing SDP (using semidefinite.SosTools)...\'', file = f)
		print('sys.stdout.flush()', file = f)
		print('B.init_sdp()', file = f)
		print('print \'SDP initialized...\'\n', file = f)
		print('print \'Minimizing SoS program...\'', file = f)
		print('sys.stdout.flush()', file = f)
		print('B.minimize()', file = f)
		print('print B.Info', file = f)
		print('print \'END\\n\'', file = f)
		print('sys.stdout.flush()', file = f, flush = True)

		print('Wrote sage script to file: ', f.name, flush = True)
	# End with
# End write_sage_script

def verify_individual_bias(svm_P, input_vars, v_1, v_2):
	n = len(input_vars)
	prot = min(i for i in range(n) if input_vars[i].is_protected == True)

	P0 = svm_P.copy()
	P1 = svm_P.copy()
	P0.substitute({prot:v_1})
	P1.substitute({prot:v_2})
	F = P0.multiply(P1)
	cons = []
	for i in range(n):
		if i != prot:
			e_i = np.zeros(shape=(n,))
			e_i[i] = 1
			cons.append(Polynomial.affine_polynomial(e_i, -input_vars[i].lower_bound))
			cons.append(Polynomial.affine_polynomial(-e_i, input_vars[i].upper_bound))
	# End for

	print('(%d, %d)' % (v_1, v_2), 'Initializing SDP...', flush = True)
	print('(%d, %d)' % (v_1, v_2), 'Time check: ', time.strftime("%H:%M:%S", time.localtime()), flush = True)
	verifier = SosVerifier(F, cons, order = 0, id = '%d_%d' % (v_1, v_2), num_jobs = NUM_JOBS)
	verifier.init_sdp()

	print('(%d, %d)' % (v_1, v_2), 'Solving SDP...', flush = True)
	print('(%d, %d)' % (v_1, v_2), 'Time check: ', time.strftime("%H:%M:%S", time.localtime()), flush = True)
	sos_min = verifier.minimize()

	print('(%d, %d)' % (v_1, v_2), 'SDP minimization complete...', flush = True)
	print('(%d, %d)' % (v_1, v_2), 'Time check: ', time.strftime("%H:%M:%S", time.localtime()), flush = True)

	print('(%d, %d)' % (v_1, v_2), 'Status: ', verifier.Info['status'])
	print(verifier.Info['Message'])
	print('(%d, %d)' % (v_1, v_2), 'SoS Lower Bound: ', sos_min)

	if sos_min is not None and sos_min >= 0.:
		return 0
	else: # TODO: Extract possible bias instance from SDP soln
		return 1
# End verify_individual_bias

def verify(svm_P, input_vars):
	print('Using %d parallel jobs...' % NUM_JOBS, flush=True)

	prot = min(i for i, v in enumerate(input_vars) if v.is_protected == True)
	prot_var = input_vars[prot]
	value_pair_list = combinations(range(int(prot_var.lower_bound), int(prot_var.upper_bound) + 1), 2)
	res = Parallel(n_jobs = NUM_JOBS, verbose = 0)(
		delayed(verify_individual_bias)(svm_P, input_vars, v_1, v_2)
		for (v_1, v_2) in value_pair_list
	)
	return res
# End verify

