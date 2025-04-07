# Author: Philips George John

import numpy as np
import cplex

from cplex.exceptions.errors import CplexSolverError

def _validate_args(P, q, A, b, C, d, lb, ub, vars):
	assert(vars is not None and len(vars) > 0)

	n = len(vars)

	assert(P is None or (P.ndim == 2 and P.shape[0] == P.shape[1] and P.shape[0] == n))
	assert(q is None or (q.ndim == 1 and q.shape[0] == n))
	assert(A is None or (A.ndim == 2 and A.shape[0] == b.shape[0] and A.shape[1] == n))
	assert(C is None or (C.ndim == 2 and C.shape[0] == d.shape[0] and C.shape[1] == n))

	assert(lb.ndim == 1 and lb.shape[0] == n)
	assert(ub.ndim == 1 and ub.shape[0] == n)

	return n
# End _validate_args

def _cplex_type(t):
	if t == 'integer':
		return 'I'
	elif t == 'boolean':
		return 'B'
	else:
		return 'C' # continuous
# End _cplex_type

def _append(a, b):
	if a is None:
		return b
	elif b is None:
		return a
	else:
		return np.concatenate((a, b), axis = 0)
# End _append

class OptimizationResult:
	SUCCESS    = 0
	INFEASIBLE = 1
	UNBOUNDED  = 2
	ERROR      = 3

	def __init__(self, status, objective = None, value = None, error = None):
		self.status = status
		self.objective = None
		self.value = None
		
		if objective is not None:
			self.objective = objective
			self.value = value
		if error is not None:
			self.error  = error
	# End __init__
# End class OptimizationResult


# Solve the following QP problem:
# Minimize     1/2 x^T P x + q^T x
# Subject to   A x <= b
#              C x  = d
#              lb <= x <= ub
#              x_i \in \mathbb{R} for all i
def cplex_solve_qp(P, q, A, b, C, d, lb, ub, vars):
	return cplex_solve_miqp(P, q, A, b, C, d, lb, ub, vars, is_miqp = False)
# End cplex_solve_qp

# Solve the following MIQP problem:
# Minimize     1/2 x^T P x + q^T x
# Subject to   A x <= b
#              C x  = d
#              lb <= x <= ub
#              x_i \in D_i (= \mathbb{R} or \mathbb{Z}, depending on vars[i].type)
def cplex_solve_miqp(P, q, A, b, C, d, lb, ub, vars, is_miqp = True):
	n      = _validate_args(P, q, A, b, C, d, lb, ub, vars)
	n_list = list(range(n))

	c = cplex.Cplex()

	c.set_log_stream(None)   # Suppress all output
	c.set_error_stream(None)
	c.set_warning_stream(None)
	c.set_results_stream(None)

	names = [v.name for v in vars]
	types = [_cplex_type(v.type) for v in vars]
	
	if is_miqp:
		c.variables.add(lb = lb.tolist(), ub = ub.tolist(), types = types, names = names)
	else:
		c.variables.add(lb = lb.tolist(), ub = ub.tolist(), names = names) # All variables are continuous (QP)

	c.objective.set_sense(c.objective.sense.minimize)

	if P is not None:
		P_list = [ [ n_list, [P[i,j] for j in range(n)] ] for i in range(n)]
		c.objective.set_quadratic(P_list)
	if q is not None:
		c.objective.set_linear([(i, q[i]) for i in range(q.shape[0])])

	if A is not None:
		A_list = [ [ n_list, [A[i,j] for j in range(n)] ] for i in range(A.shape[0])]
		c.linear_constraints.add(lin_expr = A_list, rhs = b.tolist(),
				   names = ['CL_%d' % (i + 1) for i in range(A.shape[0])],
				   senses = 'L' * A.shape[0])
	if C is not None:
		C_list = [ [ n_list, [C[i,j] for j in range(n)] ] for i in range(C.shape[0])]
		c.linear_constraints.add(lin_expr = C_list, rhs = d.tolist(),
				   names = ['CE_%d' % (i + 1) for i in range(C.shape[0])],
				   senses = 'E' * C.shape[0])

	c.solve()
	
	try:
		sol = c.solution
		status   = sol.get_status()
		s_status = sol.get_status_string()
		obj_val  = sol.get_objective_value()
		x_val    = np.array(sol.get_values())
		error    = '%s (%d)' % (s_status, status)

		if 'infeasible' in s_status:
			return OptimizationResult(OptimizationResult.INFEASIBLE, error = error)
		elif 'unbounded' in s_status:
			return OptimizationResult(OptimizationResult.UNBOUNDED, error = error)
		else:
			good_stats = [sol.status.optimal, sol.status.MIP_optimal, sol.status.feasible, sol.status.MIP_feasible]
			if status in good_stats or 'optimal' in s_status:
				return OptimizationResult(OptimizationResult.SUCCESS, objective = obj_val, value = x_val, error = error)
			else:
				return OptimizationResult(OptimizationResult.ERROR, error = error)
	
	except CplexSolverError as cse:
		if cse.args[2] == 1217:  # CPLEX Error  1217: No solution exists.
			return OptimizationResult(OptimizationResult.INFEASIBLE, error = cse.args[0])
		else:
			raise cse
	# End try
# End cplex_solve_miqp
