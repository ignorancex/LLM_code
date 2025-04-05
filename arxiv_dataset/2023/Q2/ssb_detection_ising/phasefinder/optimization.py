import cvxopt
import numpy as np

def cvxopt_solve_qp(P, q, G=None, h=None, A=None, b=None):
	assert np.array_equal(P, P.T), "P must be a symmetric matrix."
	cvxopt.solvers.options["show_progress"] = False
	args = [cvxopt.matrix(P), cvxopt.matrix(q)]
	if G is not None:
		args.extend([cvxopt.matrix(G), cvxopt.matrix(h)])
		if A is not None:
			args.extend([cvxopt.matrix(A), cvxopt.matrix(b)])
	sol = cvxopt.solvers.qp(*args)
	if 'optimal' not in sol['status']:
		return None
	return np.array(sol['x']).reshape((P.shape[1],))
