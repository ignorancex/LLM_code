from ortools.linear_solver import pywraplp
import numpy as np

from timeit import default_timer as timer

from scipy.optimize import linprog

use_gurobi_solver = False #TODO do not harcode

if use_gurobi_solver:
    import gurobipy as gp
    from gurobipy import GRB


use_cache = True #TODO do not hardcode

import collections

class LRUCache:
    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = collections.OrderedDict()


    def __contains__(self, key):
        return key in self.cache

    def get(self, key):
        try:
            value = self.cache.pop(key)
            self.cache[key] = value
            return value
        except KeyError:
            return -1

    def set(self, key, value):
        try:
            self.cache.pop(key)
        except KeyError:
            if len(self.cache) >= self.capacity:
                self.cache.popitem(last=False)
        self.cache[key] = value

    def __getitem__(self, key):
        return self.get(key)
    
    def __setitem__(self, key, value):
        self.set(key, value)

cache_dict = LRUCache(32000)

def solve_min_max(constraint_table, b_table, current_var_to_solve, n_vars, unallocated, feas_tol=1e-6, opt_tol=1e-8, current_iteration=1, use_strict_mode=False):        
    if not use_strict_mode and unallocated <= 1e-5: #-4
        #print(f"Warning: Unallocated is very small use fixed values: {unallocated} {current_var_to_solve} {n_vars} {constraint_table} {b_table}")
        return 0.0, unallocated
    
    if use_cache:
        c_hash = hash(constraint_table.data.tobytes())
        b_hash = hash(b_table.data.tobytes())
        unallocated_hash = hash(unallocated) 

        overall_hash = (c_hash, b_hash, unallocated_hash)
    
        if overall_hash in cache_dict:
            return cache_dict[overall_hash]
    
    # idx_zero = np.all(np.isclose(constraint_table, 0), axis=1)
    # b_table = b_table[~idx_zero]
    # constraint_table = constraint_table[~idx_zero]


    c = np.zeros(n_vars)
    c[current_var_to_solve] = 1


    A_eq = np.ones((1, n_vars))
    b_eq = unallocated

    res = linprog(c , A_ub=constraint_table, b_ub=b_table, A_eq=A_eq, b_eq=b_eq , bounds=(0, 1), method="highs-ds", options={"presolve": True, "primal_feasibility_tolerance": feas_tol, "dual_feasibility_tolerance": feas_tol, "ipm_optimality_tolerance": opt_tol})


    if res.success == False: #can happen due to numerical imprecisions
        min_ = 0.0
        if use_strict_mode:
            return -1, -1
    else:
        min_ = res.x[current_var_to_solve]

    if not use_strict_mode and min_ < 0.0:
        min_ = 0.0
    
    c[current_var_to_solve] = -1
    res = linprog(c , A_ub=constraint_table, b_ub=b_table, A_eq=A_eq, b_eq=b_eq , bounds=(0,1), method="highs-ds", options={"presolve": True, "primal_feasibility_tolerance": feas_tol, "dual_feasibility_tolerance": feas_tol, "ipm_optimality_tolerance": opt_tol})

    if res.success == False:
        max_ = 1e-5
        
        if use_strict_mode:
            return -1, -1
    else:
        max_ = res.x[current_var_to_solve]

    if not use_strict_mode and max_ < 0.0:
        max_ = 0.0
    
    if not use_strict_mode and max_ < min_: #can happen due to numerical imprecisions
        max_ = min_ + 1e-5
    

    if use_cache:
        cache_dict[overall_hash] = (min_, max_)

    return min_, max_


if use_gurobi_solver:
    gpenv = gp.Env(params={"OutputFlag": 0, "FeasibilityTol": 1e-2, "OptimalityTol": 1e-3})
    #gpenv = gp.Env(params={"OutputFlag": 1})

def solve_min_max_gurobi(constraint_table, b_table, current_var_to_solve, n_vars, unallocated):
    # start = timer()
    # min_, max_ = solve_min_max__(constraint_table, b_table, current_var_to_solve, n_vars, unallocated)
    # end = timer()
    # #print(end - start) # Time in seconds
    # time_ortools = end - start

    #start = timer()


    # c_hash = hash(constraint_table.data.tobytes())
    # b_hash = hash(b_table.data.tobytes())
    # unallocated_hash = hash(unallocated) 

    # overall_hash = (c_hash, b_hash, unallocated_hash)
    
    # if overall_hash in cache_dict:
    #    return cache_dict[overall_hash]

    #idx_zero = np.all(np.isclose(constraint_table, 0), axis=1)
    #b_table = b_table[~idx_zero]
    #constraint_table = constraint_table[~idx_zero]

    m = gp.Model(env=gpenv)
    
    # Create variables
    x = m.addMVar(shape=n_vars, lb=0.0, ub=1.0)
    
    m.setObjective(x[current_var_to_solve], GRB.MAXIMIZE)
    
    m.addConstr(constraint_table @ x <= b_table)
    m.addConstr(gp.quicksum(x) == unallocated)

    # Optimize model
    m.optimize()

    #print(x.X)
    #print('Obj: %g' % m.objVal)
    
    max = x[current_var_to_solve].X
    
    m.setObjective(-x[current_var_to_solve], GRB.MAXIMIZE)
        
        
    m.optimize()
    min = x[current_var_to_solve].X
    # end = timer()
    # print(end - start) # Time in seconds
    #time_scipy = end - start
    #print(time_scipy / time_ortools * 100)


    # cache_dict[overall_hash] = (min, max)
    #end = timer()
    #print(end - start) # Time in seconds

    m.dispose()
    
    return min, max





def solve_min_max__(constraint_table, b_table, current_var_to_solve, n_vars, unallocated):
    #constraint_table = [[1.0, 0.0], [0.0, 1.0]]

    #b_table = [0.6, 0.6]


    solver = pywraplp.Solver.CreateSolver("GLOP")
    #solver.SetSolverSpecificParametersAsString("use_dual_simplex: true")

    #solver = pywraplp.Solver("PDLP", pywraplp.Solver.PDLP_LINEAR_PROGRAMMING)
    #solver.SetSolverSpecificParametersAsString("termination_criteria { eps_optimal_absolute: 1e-6 eps_optimal_relative: 1e-6 }")

    #solver = pywraplp.Solver.CreateSolver("CLP")
    #solver = pywraplp.Solver.CreateSolver("GUROBI_LINEAR_PROGRAMMING")

    variables = [solver.NumVar(0.0, 1.0, "x"+str(i)) for i in range(n_vars)] #Create variables and bound them between 0 and 1

    #print("Number of variables =", solver.NumVariables())

    for row in range(len(constraint_table)):
        constraint_expr = [constraint_table[row][coefficient] * variables[coefficient] for coefficient in range(len(constraint_table[row]))]
        solver.Add(sum(constraint_expr) <= b_table[row])


    solver.Add(sum([variables[i] for i in range(n_vars)]) == unallocated) 

    #print("Number of constraints =", solver.NumConstraints())


    solver.Maximize(variables[current_var_to_solve])

    status = solver.Solve()

    if status != solver.OPTIMAL:
        print("The problem does not have an optimal solution!")
        if status == solver.FEASIBLE:
            print("A potentially suboptimal solution was found.")
        else:
            print("The solver could not solve the problem.")
            if use_gurobi_solver:
                return solve_min_max_gurobi(constraint_table, b_table, current_var_to_solve, n_vars, unallocated)
            else:
                raise RuntimeError(f"The solver could not solve the problem. {constraint_table} {b_table} {current_var_to_solve} {n_vars} {unallocated}")

    #print(f"Problem solved in {solver.wall_time():d} milliseconds")
    #print(f"Problem solved in {solver.iterations():d} iterations")

    solution_max = variables[current_var_to_solve].solution_value()
    #print(f"Max Solution for variable {current_var_to_solve} = {solution_max}")


    solver.Minimize(variables[current_var_to_solve])

    status = solver.Solve()

    if status != solver.OPTIMAL:
        print("The problem does not have an optimal solution!")
        if status == solver.FEASIBLE:
            print("A potentially suboptimal solution was found.")
        else:
            print("The solver could not solve the problem.")
            if use_gurobi_solver:
                return solve_min_max_gurobi(constraint_table, b_table, current_var_to_solve, n_vars, unallocated)
            else:
                raise RuntimeError("The solver could not solve the problem.")


    #print(f"Problem solved in {solver.wall_time():d} milliseconds")
    #print(f"Problem solved in {solver.iterations():d} iterations")

    solution_min = variables[current_var_to_solve].solution_value()
    #print(f"Min Solution for variable {current_var_to_solve} = {solution_min}")

    return solution_min, solution_max



def solve_min_max_vertices(contraint_table, b_table, current_var_to_solve, n_vars, unallocated):
    A = contraint_table
    b = b_table

    import cdd

    
    b = b.reshape((b.shape[0], 1))
    mat = cdd.Matrix(np.hstack([b, -A]), number_type="float")
    mat.rep_type = cdd.RepType.INEQUALITY
    
    mat_eq = cdd.Matrix(np.hstack([np.ones((1, n_vars)), np.ones((1, n_vars))* unallocated]), number_type="float")
    #mat_eq.rep_type = cdd.RepType.GENERATOR
    
    mat.lin_set = mat_eq
    P = cdd.Polyhedron(mat)
    g = P.get_generators()
    V = np.array(g)
    vertices = []
    for i in range(V.shape[0]):
        if V[i, 0] != 1:  # 1 = vertex, 0 = ray
            raise Exception("Polyhedron is not a polytope")
        elif i not in g.lin_set:
            vertices.append(V[i, 1:])

    if len(vertices) == 1 or len(vertices) == 0:
        raise RuntimeError("Polytope is empty")
    
    min = np.array(vertices)[:, current_var_to_solve].min()
    max = np.array(vertices)[:, current_var_to_solve].max()
    return min, max
    