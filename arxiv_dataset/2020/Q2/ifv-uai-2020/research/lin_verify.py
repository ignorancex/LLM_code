# Author: Philips George John

import sys
sys.path.insert(0, '../dev-pkgs')
sys.path.insert(0, '.')

from itertools import product

from copy import copy

import numpy as np

from ibv.model import InputVariable

import joblib
from joblib import Parallel, delayed
NUM_JOBS = 24

from lp_solve import cplex_solve_milp
from lp_solve import OptimizationResult

eps = 1e-10

def minimize_fdiff(w, bias, prot_attr, lb, ub, opt_vars):				
	# objective function: h(x) = w^T x - w^T x_1
	h = np.zeros(shape=(len(opt_vars),))
	h[0::2] = w
	h[1::2] = -w
	
	n = int(len(opt_vars)/2)
	
	# Inequality constraints
	n_ineq_constrs = 2 + 2*len([i for i in range(n) if opt_vars[2*i].pert_bound != 0. and not(opt_vars[2*i].is_protected)])
	A = np.zeros(shape=(n_ineq_constrs, len(opt_vars)))
	b = np.zeros(shape=(n_ineq_constrs,))
	
	# w^T x <= -bias (w^T x + bias <= 0)
	A[0,0::2] = w
	b[0] = -bias - eps
	
	# -w^T x_1 <= bias (w^T x_1 + bias >= 0) 
	A[1,1::2] = -w
	b[1] = bias - eps
	
	# Perturbation constraints
	c_idx = 2
	for i in range(n):
		if i not in prot_attr and opt_vars[2*i].pert_bound != 0.:
			if opt_vars[2*i].pert_bound is not None:
				pb = opt_vars[2*i].pert_bound
			else:
				pb = 2 * (opt_vars[2*i].upper_bound - opt_vars[2*i].lower_bound + 1)
			# x[i] - x_1[i] <= pb
			A[c_idx, 2*i], A[c_idx, 2*i+1] = 1, -1
			b[c_idx] = pb			
			c_idx += 1
			
			# x_1[i] - x[i] <= pb
			A[c_idx, 2*i], A[c_idx, 2*i+1] = -1, 1
			b[c_idx] = pb
			c_idx += 1
		# End if
	# End for
	
	# Equality constraints
	n_eq_constrs = len([i for i in range(n) if opt_vars[2*i].pert_bound == 0.])
	C = np.zeros(shape=(n_eq_constrs, len(opt_vars)))
	d = np.zeros(shape=(n_eq_constrs,))

	c_idx = 0
	for i in range(n):
		if opt_vars[2*i].pert_bound == 0.:
			C[c_idx, 2*i], C[c_idx, 2*i+1] = 1, -1
			c_idx += 1
		# End if
	# End for

	result = cplex_solve_milp(h, A, b, C, d, lb, ub, opt_vars)

	if result.value is not None:
		x1 = result.value[0::2]
		x2 = result.value[1::2]
		return [result.objective, x1, x2]
	else:
		return [result.objective, None, None]
	# End if
# End minimize_fij


# Verify individual bias for the linear model f(x) = sign(w^T x + b)
def verify(w, b, input_vars):
	print('Using %d parallel jobs...' % NUM_JOBS, flush=True)

	prot_attr = [i for (i, var) in enumerate(input_vars) if var.is_protected == True]
	print('Using Protected attributes:', ['%s : %d' % (input_vars[p].name, p) for p in prot_attr])
	
	# p_vals_lst = []
	# for p in prot_attr:
	# 	p_vals_lst.append(list(range(input_vars[p].lower_bound, input_vars[p].upper_bound + 1)))
	# p_vals = list(product(*p_vals_lst))

	opt_vars = []
	for var in input_vars:
		opt_vars.append(copy(var))
		opt_vars.append(InputVariable(var.name + '_1', var.type, var.lower_bound, var.upper_bound, var.pert_bound, var.is_protected))
	# End for

	lb = np.array([v.lower_bound for v in opt_vars], dtype=np.float64)
	ub = np.array([v.upper_bound for v in opt_vars], dtype=np.float64)
	
	res = []
	res.append(minimize_fdiff(w, b, prot_attr, lb, ub, opt_vars))

	fij_lower = 1e14
	bias_instance = None
	for it in res:
		if it[1] is not None:
			y1 = np.dot(w, it[1]) + b
			y2 = np.dot(w, it[2]) + b

			if (y1 >= 0 and y2 < 0) or (y1 < 0 and y2 >= 0):
				if bias_instance is None or it[0] < fij_lower:
					fij_lower = it[0]
					bias_instance = {'x': it[1], 'y': it[2]}
				# End if
			# End if
		# End if
	# End for
	
	if bias_instance is None:
		bias_instance = {}
	# End if
	
	return fij_lower, bias_instance
# End verify
