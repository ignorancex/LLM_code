# Author: Philips George John

import sys
sys.path.insert(0, '../dev-pkgs')
sys.path.insert(0, '.')

from itertools import product

from copy import copy

import numpy as np

import joblib
from joblib import Parallel, delayed
NUM_JOBS = 24

from ibv.model import InputVariable

from qp_solve import cplex_solve_miqp
from qp_solve import OptimizationResult

# TODO: We return the bias instance by raising an Exception!!
# This is the easiest way to ensure that the parallel searches are terminated.
# And AFAIK, the only possible method with the joblib abstraction.
# Find a better abstraction, even if it means using multiprocessing!!
class BiasInstanceException(Exception):
	def __init__(self, x1, x2):
		self.result = {'x':x1, 'y':x2}
	# End __init__
# End class BiasInstanceException

PROJ_VAR0 = InputVariable('_x_{n+1}', 'real', 0.0, 0.0)
PROJ_VAR1 = InputVariable('_xp_{n+1}', 'real', 0.0, 0.0)

# joblib Parallel function
def _check_local_bias(model, sv_i, sv_j, m_i, m_j, rad_i, rad_j, prot_attr, opt_vars):
	if m_i * m_j >= 0:
		return None

	eps = 1e-8 # Very small constant

	# Ensure that we can subsetquently take m_i >= 0 and m_j < 0
	# This simplifies future calculations
	if m_i < 0:
		sv_i, sv_j   = sv_j, sv_i
		m_i, m_j     = m_j, m_i
		rad_i, rad_j = rad_j, rad_i

	b_sign = np.sign(model.bias)
	neg_gamma = -1.0 * model.gamma
	b = min(
			np.abs(model.bias),
			np.abs(m_i)*np.exp(neg_gamma*rad_i),
			np.abs(m_j)*np.exp(neg_gamma*rad_j)
	)
	delta_i = min(b / (np.abs(m_i)*np.exp(neg_gamma*rad_i)), 1.0 - eps)
	delta_j = min(b / (np.abs(m_j)*np.exp(neg_gamma*rad_j)), 1.0 - eps)

	if b_sign < 0: # Negative bias
		fac_i = np.log(1.0 - delta_i) / neg_gamma
		fac_j = 0.
	elif b_sign > 0: # Positive bias
		fac_i = 0.
		fac_j = np.log(1.0 - delta_j) / neg_gamma
	else:
		fac_i = 0.
		fac_j = 0.

	sv_i = np.append(sv_i, fac_i)
	sv_j = np.append(sv_j, fac_j)

	# Viable candidates
	z_list = SVMWrapper._search_line(sv_i, sv_j, rad_i, rad_j, prot_attr, opt_vars)
	for z in z_list:
		x1 = z[0::2][:-1] # Remove extra var added by projection
		x2 = z[1::2][:-1]
		if model.check_bias_instance(x1, x2):
			raise BiasInstanceException(x1, x2)
	return None
# End _check_local_bias

class SVMWrapper:
	def __init__(self, support_vectors, multipliers, bias, kernel = 'rbf', C = 1., **params):
		self.kernel   = kernel
		self.C        = C
		self.sv_x     = support_vectors
		self.m        = np.ravel(multipliers)
		self.bias     = bias
		self.params   = params
		self.gamma    = self.params.setdefault('gamma', 0.5)

		assert(self.sv_x.shape[0] == self.m.shape[0])

		variance_param =  np.abs(1.0 / (2*self.gamma))
		self.DM, self.radii, self.sv_weights = SVMWrapper._compute_metrics(self.sv_x, self.m, variance_param)

#		self.kernel_func = self._get_kernel_func() # This causes pickle errors with joblib
	# End __init__

#	def _get_kernel_func(self):
#		gamma = self.gamma
#		if self.kernel == 'rbf':
#			def rbf_kernel(x_1, x_2):
#				return np.exp(-1. * gamma * np.sum(np.square(x_1 - x_2), axis = -1))
#			return rbf_kernel
#		else: # Linear kernel
#			self.kernel = 'linear'
#			def lin_kernel(x_1, x_2):
#				return np.dot(x_1, x_2)
#			return lin_kernel
#	# End _get_kernel_func

	def rbf_kernel(self, x_1, x_2):
		return np.exp(-1. * self.gamma * np.sum(np.square(x_1 - x_2), axis = -1))
	# End rbf_kernel

	# Compute the following metrics to be used in searching the feature space:
	#    1. The pairwise L_2 distance matrix between the centers (support vectors)
	#    2. The radius of the L_infty ball to consider for each center
	#    3. The 'weight' of each center (used to determine search order)
	def _compute_metrics(centers, multipliers, variance_param):
		assert(centers.ndim == 2 and multipliers.ndim == 1)
		assert(centers.shape[0] == multipliers.shape[0])
		M = centers.shape[0]
		eps = 1e-8

		distances = []
		for i in range(centers.shape[0]):
			row = np.linalg.norm(centers - centers[i], ord = 2, axis = 1)
			distances.append(row)
		dist_matrix   = np.array(distances)

		gamma          = 1.0 / (2*variance_param)
		four_sigma     = 4.0 * np.sqrt(variance_param)
		radii          = np.ceil(four_sigma * np.sqrt(np.log((M/eps) * np.abs(multipliers))))

		mult_wts       = np.exp(-gamma * np.square(dist_matrix))
		center_weights = np.ravel(np.matmul(mult_wts, np.expand_dims(multipliers, axis = 1)))

	#	center_weights = center_weights + (1./np.sqrt(variance_param)) * np.sign(center_weights) * radii
		center_weights += np.sign(center_weights) * radii

		return dist_matrix, radii, center_weights
	# End _compute_metrics

	# Returns True if the l_infty balls B(c_1, rad_1) and B(c_2, rad_2)
	# intersect (possibly at the boundary) in all axes except [prot].
	# And False otherwise.
	def _strong_intersect_linfty(c_1, c_2, rad_1, rad_2, prot):
		assert(c_1.ndim == 1 and c_2.ndim == 1)
		assert(c_1.shape[0] == c_2.shape[0])
		for i in range(c_1.shape[0]):
			if i not in prot:
				if (c_1[i] + rad_1) < (c_2[i] - rad_2) or (c_2[i] + rad_2) < (c_1[i] - rad_1):
					return False
		# End for
		return True
	# End _strong_intersect_linfty

	def _search_line(sv_i, sv_j, rad_i, rad_j, prot, opt_vars):
		assert(len(opt_vars) == 2*sv_i.shape[0])

		if not SVMWrapper._strong_intersect_linfty(sv_i, sv_j, rad_i, rad_j, prot):
			return []

		n = int(len(opt_vars) / 2)
		out = []

		# QP FORMULATION:
		# Minimize     1/2 x^T P x + q^T x
		# Subject to   A x <= b
		#		       C x  = d
		#              lower_bound(x_i) <= x_i <= upper_bound(x_i)

		# Optimization Variables are denoted (x_1, xp_1, x_2, xp_2, ...., x_n, xp_n)

		# Objective: Minimize 1/2 (|| (x_{-p}, pv_1) - sv_i ||^2 + || (x_{-p}, pv_2) - sv_j ||^2)
		# We omit the constant term obtained after expanding
		# the norms (since it does not make any difference to the minimizer)
		P = np.eye(2*n)
		q = np.zeros(shape=(2*n,))
		for k in range(n):
			q[2*k], q[2*k+1] = -sv_i[k], -sv_j[k]

		# Equality Constraints [Cx = d] (n - 1 constraints)
		C = np.zeros(shape=(n - 1,2*n)) # Equality
		d = np.zeros(shape=(n - 1,))

		# x_k - xp_k = 0 for all k not in [prot]
		ic = 0
		for k in range(n):
			if k not in prot:
				C[ic][2*k], C[ic][2*k+1] = 1, -1
				d[ic] = 0
				ic += 1
		# End for

		# Upper and lower bounds
		lb = np.zeros(shape = (2*n,))
		ub = np.zeros(shape = (2*n,))
		for k in range(n):
			lb[2*k]   = max(opt_vars[2*k].lower_bound, sv_i[k] - rad_i)
			ub[2*k]   = min(opt_vars[2*k].upper_bound, sv_i[k] + rad_i)
			lb[2*k+1] = max(opt_vars[2*k+1].lower_bound, sv_j[k] - rad_j)
			ub[2*k+1] = min(opt_vars[2*k+1].upper_bound, sv_j[k] + rad_j)
		# End for

		# Inequality Constraints [Ax <= b] (1 constraint)
		A = np.zeros(shape=(1,2*n))
		b = np.zeros(shape=(1,))

		# One direction : sum(pr in prot) x_pr - xp_pr <= -1
		for pr in prot:
			A[0][2*pr], A[0][2*pr+1] = 1, -1
		b[0] = -1

		try:
			result = cplex_solve_miqp(P, q, A, b, C, d, lb, ub, opt_vars)
			if result.status == OptimizationResult.SUCCESS:
				out.append(result.value)
			else:
				print('OptimizationResult: Status = %d, Error = %s' % (result.status, result.error), flush = True)
		except Exception as ex:
			print('OptimizationResult: Exception :: %s ' % ex, flush = True)

		# One direction : sum(pr in prot) x_pr - xp_pr >= 1
		for pr in prot:
			A[0][2*pr], A[0][2*pr+1] = -1, 1
		b[0] = -1
		try:
			result = cplex_solve_miqp(P, q, A, b, C, d, lb, ub, opt_vars)
			if result.status == OptimizationResult.SUCCESS:
				out.append(result.value)
			else:
				print('OptimizationResult: Status = %d, Error = %s' % (result.status, result.error), flush = True)
		except Exception as ex:
			print('OptimizationResult: Exception :: %s' % ex, flush = True)

		return out
	# End _search_line

	def project(self, X):
		IP = np.zeros(shape = (X.shape[0],), dtype = np.float32)

		for i in range(self.m.shape[0]):
			IP += self.m[i] * self.rbf_kernel(X, self.sv_x[i])

		return (IP + self.bias)
	# End project

	def predict(self, X):
		return (self.project(X) > 0).astype(np.int32)
	# End predict

	def check_bias_instance(self, x_1, x_2):
		X = np.array([x_1, x_2])
		y = self.predict(X)
		return y[0] != y[1]
	# End check_bias_instance

	def verify_ind_bias(self, input_vars, is_relaxation = True):
		print('Using %d parallel jobs...' % NUM_JOBS, flush=True)

		prot_attr = [i for (i, var) in enumerate(input_vars) if var.is_protected == True]
		print('Using Protected attributes:', ['%s : %d' % (input_vars[p].name, p) for p in prot_attr])
		p_vals_lst = []
		for p in prot_attr:
			p_vals_lst.append(list(range(input_vars[p].lower_bound, input_vars[p].upper_bound + 1)))
		p_vals = product(*p_vals_lst)
		
		opt_vars = [] # In our optimization problem, each var appears twice (denoted x and xp)
		for v in input_vars:
			opt_vars.append(v)
			v1 = copy(v)
			v1.name = v1.name + '_1'
			opt_vars.append(v1)
		# End for

		# Add projection (extra-dim) variables for handling bias term
		opt_vars.append(PROJ_VAR0)
		opt_vars.append(PROJ_VAR1)

		if self.kernel == 'linear':
			raise RuntimeError('Please use the ibv.linear package to verify linear SVM models.')
		elif self.kernel == 'rbf':
			idxs_n = self.sv_weights.argsort()
			idxs_p = (-self.sv_weights).argsort()
			radii_p, mult_p, centers_p = self.radii[idxs_p], self.m[idxs_p], self.sv_x[idxs_p]
			radii_n, mult_n, centers_n = self.radii[idxs_n], self.m[idxs_n], self.sv_x[idxs_n]
#			mult, centers = shuffle(self.m, self.sv_x, random_state = _rs())

			n_sv = centers_p.shape[0]
			try:
				# For debugging exceptions
				#for pv_1, pv_2 in product(p_vals, p_vals):
				#	for i, j in product(range(n_sv), range(n_sv)):
				#		_check_local_bias(self, pv_1, pv_2, centers_p[i], centers_n[j], mult_p[i], mult_n[j], radii_p[i], radii_n[j], prot_attr, opt_vars, is_relaxation)
				# End for
				Parallel(n_jobs = NUM_JOBS)(delayed(_check_local_bias)(
						self, centers_p[i], centers_n[j], mult_p[i], mult_n[j],
						radii_p[i], radii_n[j], prot_attr, opt_vars
					) for i, j in product(range(n_sv), range(n_sv))
				)
			# NB: Horrible abstraction, but easiest to get working for early termination.
			# TODO: Change this to something better.
			except BiasInstanceException as bie:
				return bie.result
			return {} # No bias
	# End verify_ind_bias
# End class SVMWrapper
