import numpy as np
import cvxpy as cp
#from cvxpylayers.torch import CvxpyLayer


import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def projection_solver_layer_lp(c, G, h, n_feat, C=None, d=None, init_x=None):
    x = cp.Variable(n_feat)
    constraints = [G @ x <= h]
    objective = cp.Minimize(c.T @ x)
    problem = cp.Problem(objective, constraints)
    assert problem.is_dpp()

    cvxpylayer = CvxpyLayer(problem, parameters=[], variables=[x])

    return cvxpylayer
# solve the problem

def solve_lp(c, G=None, h=None, C=None, d=None, init_x=None, n_jobs=1, solver=cp.GUROBI):
    n = len(c)
    options = {'threads': n_jobs}
    x = cp.Variable(shape=(n, 1))
    obj = cp.Minimize(c.T @ x)
    if C is not None and d is not None and G is not None and h is not None:
        constraints = [G@x <= h, C@x == d]
    elif C is not None and d is not None:
        constraints = [C@x == d]
    else:
        constraints = [G@x <= h]
    prob = cp.Problem(obj, constraints)
    if init_x is not None:
        x.value = init_x
        prob.solve(solver=solver, warm_start=True, **options)
    else:
        prob.solve(solver=solver, **options)
    return prob.status, x.value

def solve_qp(Q, q, G, h, n, C=None, d=None, init_x=None, n_jobs=1, solver=cp.GUROBI):
    options = {'threads': n_jobs}
    x = cp.Variable(shape=(n, 1))
    obj = cp.Minimize((1/2)*cp.quad_form(x, Q) + q.T @ x)

    if C is not None and d is not None:
        constraints = [G@x <= h, C@x == d]
    else:
        constraints = [G@x <= h]
    prob = cp.Problem(obj, constraints)

    if init_x is not None:
        x.value = init_x

    try:
        prob.solve(solver=solver, **options)
    except cp.error.SolverError:
        try:
            prob.solve(solver=solver, **options)
        except cp.error.SolverError as e:
            print(e)
            logger.error("Rare")
            return False, x.value
    return prob.status, x.value