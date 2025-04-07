# Author: Philips George John

import math
import numpy as np
from sdp import sdp

from polynomial import Polynomial


def Matrix(n, v):
	return v * np.eye(n)
# End Matrix


# By definition, the Local Moment Matrix (for a poly constraint g),
# with first moment degree d_1, is contructed as follows:
# n = num_vars, and g is a polynomial in n variables
# v = [all monomials in n-vars with deg <= d, listed in some ordering]
# M = v^T v, is the second moment matrix of monomials
# LMM = g * M (where we multiply each monomial in M by the polynomial g)
#
# This class can be used to find the coefficient of a monomial M
# in LMM[i, j], which is what we need for the SoS SDP construction.
# However, it does not construct the LMM as such, for efficiency
# considerations.
class LocalMomentMatrix:
	def __init__(self, num_vars, g, first_moment_deg):
			self.num_vars = num_vars
			self.g = g
			self.first_moment_deg = first_moment_deg

			self.monomials = list(Polynomial(num_vars, first_moment_deg).monomials())
			self.moment_mat_size = len(self.monomials)
	# End __init__

	# i, j \in {0, ..., moment_mat_size - 1}
	# mon is a monomial in std rep
	# Returns the coefficient of mon in LMM[i, j] (see above)
	def coef(self, i, j, mon):
		assert(i >= 0 and i < self.moment_mat_size)
		assert(j >= 0 and j < self.moment_mat_size)

		# Let M_i denote the i-th monomial of R
		# The {ij}-th entry of the local moment matrix will be
		# g * M_ij
		# Where M_ij = M_i * M_j
		M_ij = Polynomial.monomial_multiply(self.monomials[i], self.monomials[j])

		# H satisfies M_ij * H = mon (in std rep)
		# So, finding the coefficient of mon in LMM[i, j]
		# is equivalent to finding the coefficient of H in g
		H = Polynomial.monomial_factor(mon, M_ij)

		# Note: If there is no such H, Polynomial.monomial_factor()
		# will return None, in which case, checking the coefficient of H = None
		# in g will return 0.
		return self.g.coef_[self.g.index(H)]
	# End coef
# End LocalMomentMatrix


class SosVerifier:
	def __init__(self, f, cons, order = 0, solver = 'csdp', id = '1', num_jobs = 4):
		self.f        = f
		self.cons     = cons
		self.order    = order
		self.solver   = solver
		self.id       = id
		self.num_jobs = num_jobs

		self.num_vars = f.num_vars()
		for g in cons:
			assert (g.num_vars() == self.num_vars)

		self.f_degree = f.degree()
		self.f_half_degree = int(math.ceil(self.f_degree / 2.))

		# Add a dummy constraint - the constant polynomial '1'
		self.cons.append(
				Polynomial.affine_power_polynomial([0] * self.num_vars, 1, 1, d = 0)
		)
		self.cons_degrees      = []
		self.cons_half_degrees = []

		for g in self.cons:
			self.cons_degrees.append(g.degree())
			self.cons_half_degrees.append(int(math.ceil(g.degree() / 2.)))
		# End for

		cns_half_deg_max = max(self.cons_half_degrees)

		# The degree of the SoS relaxation to be used
		self.relaxation_degree = max(1, self.f_half_degree, cns_half_deg_max) + self.order

		self.monomials = list(Polynomial(self.num_vars, 2*self.relaxation_degree).monomials())
	# End __init__

	def full_polynomial_coefs(self):
		c = np.zeros(shape = (len(self.monomials),))
		for i, M in enumerate(self.monomials):
			j = self.f.index(M)
			c[i] = self.f.coef_[j]
		# End for
		return c
	# End full_polynomial_coefs

	def local_moment_matrix_slice(self, i, mon, LMM):
		C = np.zeros(shape = (LMM.moment_mat_size, LMM.moment_mat_size))
		for i in range(C.shape[0]):
			for j in range(i, C.shape[1]):
				C[i,j] = LMM.coef(i, j, mon)
				C[j,i] = C[i,j]
			# End for
		# End for
		return i, C
	# End local_moment_matrix_slice

	def init_sdp(self):
		n = self.num_vars
		N = Polynomial.num_monomials(n, 2*self.relaxation_degree)
		SDP_A = [[] for i in range(N)]
		SDP_C = []
		SDP_b = self.full_polynomial_coefs()

		mat_sz = Polynomial.num_monomials(n, self.relaxation_degree)
		print('SOS_SDP: n = %d, rel_deg = %d, sdp_vars = %d, sdp_mat_sz = %d' % (n, self.relaxation_degree, N, mat_sz))
		for idx in range(len(self.cons)):
			fm_deg = self.relaxation_degree - self.cons_half_degrees[idx]
			d = Polynomial.num_monomials(n, fm_deg)
			h = np.zeros(shape = (d, d))
			SDP_C.append(h)

			LMM = LocalMomentMatrix(self.num_vars, self.cons[idx], fm_deg)
#			A_list = Parallel(n_jobs = self.num_jobs, verbose = 0)(
#				delayed(self.local_moment_matrix_slice)(i, M, LMM)
#				for i, M in enumerate(self.monomials)
#			)
			for i, M in enumerate(self.monomials):
				_, A = self.local_moment_matrix_slice(i, M, LMM)
				SDP_A[i].append(A)
			# End for
		# End for

		SDP_A[0].append(Matrix(1, 1.))
		SDP_A[0].append(Matrix(1, -1.))
		for i in range(1, N):
			SDP_A[i].append(Matrix(1, 0.))
			SDP_A[i].append(Matrix(1, 0.))
		# End for

		SDP_C.append(Matrix(1, 1.))
		SDP_C.append(Matrix(1, -1.))

		self.SDP_b = SDP_b
		self.SDP_A = SDP_A
		self.SDP_C = SDP_C
	# End init_sdp

	def minimize(self):
		sos_sdp = sdp(solver = 'csdp', id = self.id)
		sos_sdp.SetObjective(self.SDP_b)
		sos_sdp.AddConstantBlock(self.SDP_C)
		for i in range(len(self.SDP_A)):
			sos_sdp.AddConstraintBlock(self.SDP_A[i])
		# End for

		sos_sdp.solve()
		if sos_sdp.Info['Status'] == 'Optimal':
			self.f_min = min(sos_sdp.Info['PObj'], sos_sdp.Info['DObj'])
			self.Info = {"min":self.f_min, "Wall":sos_sdp.Info.get('Wall'), "CPU":sos_sdp.Info.get('CPU')}
			self.Info['status'] = 'Optimal'
			self.Info['Message'] = 'Feasible solution for SoS of order ' + str(self.relaxation_degree)
		else:
			self.f_min = None
			self.Info = {"min":None}
			self.Info['status'] = 'Infeasible'
			self.Info['Message'] = 'No feasible solution for SoS of order ' + str(self.relaxation_degree)
		return self.f_min
	# End minimize
# End class SosVerifier


## ###################################
##   SANITY CHECK FOR SOS SDP BOUNDS
## ###################################
if __name__ == '__main__':
	print('Initializing SDP...')
	n = 20
	v = np.random.randint(1, 20, n)
	f = Polynomial.affine_power_polynomial(v, gamma = 1, r = 1, d = 2)
	cons = []
	for i in range(n):
		# Add constraints 0 <= x_i <= 1, in {g >= 0} form
		e_i = np.zeros(shape=(n,))
		e_i[i] = 1
		cons.append(Polynomial.affine_polynomial(a = e_i, b = -0))
		cons.append(Polynomial.affine_polynomial(a = -e_i, b = 1))
	# End for

	verifier = SosVerifier(f, cons, order = 1)
	verifier.init_sdp()

	print('Solving SDP...')
	sos_min = verifier.minimize()

	print('Status: ', verifier.Info['status'])
	print(verifier.Info['Message'])
	print('SoS Lower Bound: ', sos_min)
	print('END')
# End main
