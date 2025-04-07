# Author: Philips George John

import numpy as np
import cplex

from cplex.exceptions.errors import CplexSolverError


def _validate_args(h, A, b, C, d, lb, ub, vars):
	assert(vars is not None and len(vars) > 0)

	n = len(vars)
	assert(h.ndim == 1 and h.shape[0] == n)
	assert(A is None or (A.ndim == 2 and A.shape[0] == b.shape[0] and A.shape[1] == n))
	assert(C is None or (C.ndim == 2 and C.shape[0] == d.shape[0] and C.shape[1] == n))

	assert(lb is None or (lb.ndim == 1 and lb.shape[0] == n))
	assert(ub is None or (ub.ndim == 1 and ub.shape[0] == n))

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

# Solve the following LP problem:
# Minimize      h^T x
#   Subject to   A x <= b
#		         C x  = d
#                lb[i] <= x_i <= ub[i]
#                x_i \in \mathbb{R} for all i
def cplex_solve_lp(h, A, b, C, d, lb, ub, vars):
	return cplex_solve_milp(h, A, b, C, d, lb, ub, vars, is_milp = False)
# End cplex_solve_qp

# Solve the following MILP problem:
# Minimize      h^T x
#   Subject to   A x <= b
#		         C x  = d
#                lb[i] <= x_i <= ub[i]
#                x_i \in D_i (= \mathbb{R} or \mathbb{Z}, depending on vars[i].type)
def cplex_solve_milp(h, A, b, C, d, lb, ub, vars, is_milp = True):
	n      = _validate_args(h, A, b, C, d, lb, ub, vars)
	n_list = list(range(n))

	c = cplex.Cplex()

	c.set_log_stream(None)   # Suppress all output
	c.set_error_stream(None)
	c.set_warning_stream(None)
	c.set_results_stream(None)

	names = [v.name for v in vars]
	types = [_cplex_type(v.type) for v in vars]
	if lb is None:
		lb = []
	if ub is None:
		ub = []
	if is_milp:
		c.variables.add(lb = lb, ub = ub, types = types, names = names)
	else:
		c.variables.add(lb = lb, ub = ub, names = names) # All variables are continuous (LP)

	c.objective.set_sense(c.objective.sense.minimize)

	c.objective.set_linear([(i, h[i]) for i in range(h.shape[0])])

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
# End cplex_solve_milp
