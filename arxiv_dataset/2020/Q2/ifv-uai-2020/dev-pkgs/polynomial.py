# Author: Philips George John

import numpy as np

from itertools import combinations_with_replacement
from scipy.special import factorial, binom


def _multinomial_coeff(c):
	return factorial(c.sum()) / factorial(c).prod()
# End multinomial_coeff


# The polynomial class provides methods for representing and manipulating
# polynomials in n indeterminates (x_1,...,x_n) and with degree <= d.
#
# This class uses two representations for monomials:
#   The **standard representation** is as a n-tuple M of integers
#      with 0 <= M[i] <= d for all i = {1,...,n}
#   M = (m_1,...,m_n) represents the monomial x_1^{m_1} ... x_n^{m_n}
#      * This is the common reprsentation used in maths.
#      * The advantage is that each monomial has a unique representation.
#      * The disadvantage is that not all such tuples will be valid monomials
#          (for that, we need the sum of the tuple entries to be <= d)
#   The **dual representation** is as a t-tuple M of integers (with t <= d)
#      with 0 <= M[i] < n for all i = {1,...,t}
#   M = (m_1,...,m_t) represents the degree-t monomial x_{m_1+1} ... x_{m_t+1}
#      * The advantage is that all such tuples will be valid monomials.
#      * Another advantage is that it is easy to generate such tuples
#           in natural monomial orderings (grlex, grevlex).
#      * The disadvantage is that a monomial will have multiple representations
#          (all t-tuples which are permutations of each other represent the same monomial)
#
#  The upshot of the above discussion is that the **standard** representation will always be used
#    externally -- when the user specifies a monomial, or a (set of) monomial(s) is returned.
#  On the other hand, the **dual** representation will be used internally, especially in operations
#    that involve enumeration of monomials.
#
#
# The polynomial coefficients will be stored as a vector 'coef_' in the 'grevlex' order (aka. 'degrevlex').
# The coefficients are sorted by degree (of the monomials),
# and by reverse lexicographic order of the monomials of the same degree
# e.g. (1, x_n, ... , x_1, x_n^2, x_{n-1} x_n, x_{n-1}^2, ... , x_2 x_n, ..., x_2^2, x_1 x_n, ..., x_1^2)
#        when d = 2
# The corresponding monomial will be stored in the dual rep. in 'dual_mons_'.

class Polynomial:
	# n = num_indeterminates, d = max degree
	def __init__(self, n, d):
		self.n = n
		self.d = d
		self.dual_mons_ = self._enum_monomials()
		self.dual_mons_.append(None)
		self.coef_      = [0.] * (len(self.dual_mons_))
		self.coef_[-1]  = 0.
	# End __init__

	@staticmethod
	def from_coefs(n, d, coefs):
		assert (len(coefs) == int(binom(n + d, d)) or len(coefs) == (int(binom(n + d, d)) + 1))
		P = Polynomial(n, d)
		P.coef_ = coefs.copy()
		if len(P.coef_) == int(binom(n + d, d)):
			P.coef_.append(0.) # Want P.ceof_[-1] = 0.
		else:
			P.coef_[-1] = 0.
		return P
	# End from_coefs

	def copy(self):
		R = Polynomial.from_coefs(self.n, self.d, self.coef_)
		return R
	# End copy

	def evaluate(self, x):
		val = 0.
		for i, M in enumerate(self.dual_mons_):
			if self.coef_[i] != 0.:
				term = self.coef_[i]
				for v in M: term = term * x[v]
				val += term
		# End for
		return val
	# End evaluate

	# Polynomial addition
	# The result uses the ordering of the higher degree
	# polynomial (among {self, Q})
	def add(self, Q):
		assert(self.n == Q.n)
		P1, P2 = self, Q
		if P1.d < P2.d:
			P1, P2 = P2, P1
		R = P1.copy()
		for i, M in enumerate(P2.dual_mons_):
			j = R._dual_mon_index(M)
			R.coef_[j] += P2.coef_[i]
		# End for
		return R
	# End add

	def negate(self):
		C = - np.array(self.coefs_)
		R = Polynomial.from_coefs(self.n, self.d, C.tolist())
		return R
	# End negate

	# Extremely naive! implementation of poly multiplication
	def multiply(self, Q):
		assert(self.n == Q.n)
		R = Polynomial(self.n, self.d + Q.d)
		for i, M_i in enumerate(self.dual_mons_):
			if self.coef_[i] != 0.:
				for j, M_j in enumerate(Q.dual_mons_):
					if Q.coef_[j] != 0.:
						M = M_i + M_j # concatenates the tuples
						r = R._dual_mon_index(M)
						R.coef_[r] += self.coef_[i] * Q.coef_[j]
				# End for
		# End for
		return R
	# End multiply

	def degree(self):
		return self.d
	# End degree

	def num_vars(self):
		return self.n
	# End num_vars

	def monomial(self, index):
		return self._mon_d2s(self.dual_mons_[index])
	# End monomial

	def monomials(self):
		return (self._mon_d2s(M) for M in self.dual_mons_[:-1])
	# End monomials

	# Return the index (in the coef_ array) of monomial M (std rep)
	def index(self, M):
		return self._dual_mon_index(Polynomial._mon_s2d(M))
	# End index

	def substitute(self, substitutions):
		m = len(substitutions)
		assert(m <= self.n)
		for ind, val in substitutions.items():
			assert(ind >= 0 and ind < self.n)
			for i, M in enumerate(self.dual_mons_):
				if self.coef_[i] != 0.:
					if ind in M:
						new_coef = self.coef_[i]
						while ind in M:
							new_coef *= val
							M = self._remove_mon_ind(M, ind)
						# End while
						i_new = self._dual_mon_index(M)
						self.coef_[i] = 0. # This monomial is no longer present
						self.coef_[i_new] += new_coef
					# End if
				# End if
			# End for
		# End for
	# End substitute

	def remap(self, permutation):
		assert(len(permutation) == self.n)
		for i, M in enumerate(self.dual_mons_):
			if M is not None:
				M_new = Polynomial._sorted_permute(M, permutation)
				self.dual_mons_[i] = M_new
		# End for
		self._standardize()
	# End remap

	# Return the polynomial object for P(x) = a^T x + b
	@staticmethod
	def affine_polynomial(a, b):
		return Polynomial.affine_power_polynomial(a, gamma = 1, r = b, d = 1)
	# End affine_polynomial

	# Return the polynomial object for P(x) = (gamma * v^T x + r)^d
	@staticmethod
	def affine_power_polynomial(v, gamma, r, d):
		v = np.array(v)
		v = gamma * v # Scale v by gamma - bilinearity
		n = v.shape[0]

		P = Polynomial(n, d)
		for i, M in enumerate(P.dual_mons_):
			if M is not None:
				k = len(M) # degree of the monomial M
				coef = r**(d - k) * Polynomial._aff_pow_monomial_count(M, n, d)
				for j in range(k): coef *= v[M[j]]
				P.coef_[i] = coef
		# End for
		return P
	# End affine_power_polynomial

	# Return a string representation of the polynomial
	def str(self):
		poly_parts = []
		for i, M in enumerate(self.dual_mons_):
			if self.coef_[i] != 0.:
				M_str = Polynomial.monomial_str(self._mon_d2s(M))
				if M_str != '1': # constant term
					poly_parts.append('%.3e %s' % (self.coef_[i], M_str))
				else:
					poly_parts.append('%.3e' % self.coef_[i])
		# End for
		return ' + '.join(poly_parts)
	# End str

	# Return the string rep. of a monomial (a_1,...,a_n) - in *std form*
	@staticmethod
	def monomial_str(M):
		if M is None: return 'None'
		M_parts = []
		for i in range(len(M)):
			if M[i] > 1:
				M_parts.append('x_%d^%d' % (i + 1, M[i]))
			elif M[i] == 1:
				M_parts.append('x_%d' % (i + 1))
		# End for
		if len(M_parts) == 0:
			M_parts.append('1')
		return ' '.join(M_parts)
	# End monomial_str

	# Get a string representation in Sage syntax
	# Assumes 'var' (default = 'x') is the list
	# of generators of the polynomial ring.
	def sage_declaration(self, var = 'x'):
		poly_parts = []
		for i, M in enumerate(self.dual_mons_):
			if self.coef_[i] != 0.:
				M = self._mon_d2s(M)
				M_parts = []
				for j in range(len(M)):
					if M[j] > 1:
						M_parts.append('%s[%d]^%d' % (var, j, M[j]))
					elif M[j] == 1:
						M_parts.append('%s[%d]' % (var, j))
				# End for
				M_str = '*'.join(M_parts)

				if len(M_str) > 0: # non constant term
					poly_parts.append('%.24f*%s' % (self.coef_[i], M_str))
				else:
					poly_parts.append('%.24f' % (self.coef_[i]))
		# End for
		return ' + '.join(poly_parts)
	# End sage_declaration

	# Return the monomial H (std rep) such that M = M_fac * H,
	# or none if no such H exists (M is not a multiple of M_fac)
	@staticmethod
	def monomial_factor(M, M_fac):
		if M is None: return None
		elif M_fac is None: return M

		assert (len(M_fac) == len(M))
		L = list(M)
		for i in range(len(M)):
			L[i] -= M_fac[i]
			if L[i] < 0: return None
		# End for
		return tuple(L)
	# End monomial_factor

	# Return M1 * M2 (std rep)
	@staticmethod
	def monomial_multiply(M1, M2):
		if M1 is None: return M2
		elif M2 is None: return M1

		assert (len(M1) == len(M2))
		L = list(M1)
		for i in range(len(M1)):
			L[i] += M2[i]
		# End for
		return tuple(L)
	# End monomial_multiply

	# Return the number of distinct monomials of degree <= d
	# with n variables
	@staticmethod
	def num_monomials(n, d):
		return int(binom(n + d, d))
	# End num_monomials

	###############################################
	############## INTERNAL FNS ###################
	###############################################

	# Given a sorted tuple (t_1,...t_m), with 0 <= t_i < n (integers)
	# and perm, a permutation of {0, 1, ..., n - 1}
	# return the tuple sorted(perm[t_1],....,perm[t_m])
	@staticmethod
	def _sorted_permute(T, perm):
		L = list(T)
		for i in range(len(L)):
			L[i] = perm[L[i]]
		L.sort()
		return tuple(L)
	# End _sorted_permute

	# Put the monomials and coefficients back in the
	# standard order after doing a remap
	def _standardize(self):
		R = Polynomial(self.n, self.d)
		for i, M in enumerate(R.dual_mons_):
			j = self._dual_mon_index(M)
			R.coef_[i] = self.coef_[j]
		# End for
		self.dual_mons_ = R.dual_mons_
		self.coef_ = R.coef_
	# End _standardize

	# Given a monomial M in *dual rep*
	# Return the number of times the monomial will appear
	# in the expansion of (1 + x_1 + ... + x_n)^d
	@staticmethod
	def _aff_pow_monomial_count(M, n, d):
		if M is None: return 0
		k = [0] * (n + 1)
		k[n] = d - len(M)
		assert(k[n] >= 0)
		for i in range(len(M)):
			k[M[i]] += 1
		# Compute and return the multinomial coefficient
		return _multinomial_coeff(np.array(k))
	# End _aff_pow_monomial_count

	# This function determines the monomial order used by the class
	# It returns *dual reps* of the monomials in the appropriate
	# sequence
	# The current implementation uses 'grevlex' increasing order.
	def _enum_monomials(self):
		n_list = list(range(self.n))
		reverse_perm = list(reversed(n_list))
		L = []
		L.append(())
		for k in range(1, self.d + 1): # Enumerate by increasing degree
			# Get list of deg-k monomials (dual rep.) in lex. order
			Mons_k = list(combinations_with_replacement(n_list, k))
			for i in range(len(Mons_k)):
				# Remap the vars (equiv. to reversing the standard rep of monomial)
				# to get them in reverse lex. order
				Mons_k[i] = Polynomial._sorted_permute(Mons_k[i], reverse_perm)
			L.extend(Mons_k)
		# End for
		return L
	# End _enum_monomials

	def _dual_mon_index(self, M):
		if M is None: return -1
		M = tuple(sorted(M))
		try:
			return self.dual_mons_.index(M)
		except ValueError:
			return -1
	# End _dual_mon_index

	def _remove_mon_ind(self, M, ind):
		if M is None: return None
		L = list(M)
		L.remove(ind)
		return tuple(sorted(L))
	# End _remove_mon_ind

	@staticmethod
	def _mon_s2d(M):
		if M is None: return None
		M_d = []
		for i in range(len(M)):
			for j in range(M[i]):
				M_d.append(i)
		return tuple(M_d)
	# End _mon_s2d

	def _mon_d2s(self, M):
		if M is None: return None
		M_s = [0] * self.n
		for i in range(len(M)):
			M_s[M[i]] += 1
		return tuple(M_s)
	# End _mon_d2s

	# Return the monomial H (dual rep) such that M = M_fac * H,
	# or none if no such H exists (M is not a multiple of M_fac)
	@staticmethod
	def _dual_monomial_factor(M, M_fac):
		if M is None: return None
		elif M_fac is None: return M

		L = list(M)
		for i in range(len(M_fac)):
			if M_fac[i] not in L:
				return None
			L.remove(M_fac[i])
		# End for
		return tuple(sorted(L))
	# End _dual_monomial_factor

	# Return M1 * M2 (dual rep)
	@staticmethod
	def _dual_monomial_multiply(M1, M2):
		if M1 is None: return M2
		elif M2 is None: return M1

		L = M1 + M2
		return tuple(sorted(L))
	# End monomial_multiply

	def __str__(self):
		return self.str()
	# End __str__
#End class Polynomial
