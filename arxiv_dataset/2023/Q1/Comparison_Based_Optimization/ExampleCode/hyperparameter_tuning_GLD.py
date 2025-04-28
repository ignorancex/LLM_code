# figure out optimal values for R & r in the GLD algorithm.

"""
experimenting with tuning GLD hyperparameters.
For PyCutest functionality, add this to ENVIRONMENT VARIABLES (edit configurations):
PYCUTEST_CACHE=.;CUTEST=/usr/local/opt/cutest/libexec;MYARCH=mac64.osx.gfo;SIFDECODE=/usr/local/opt/sifdecode/libexec;MASTSIF=/usr/local/opt/mastsif/share/mastsif;ARCHDEFS=/usr/local/opt/archdefs/libexec
"""

# INSTRUCTIONS:
# choose 3 PyCutest problems.
# run GLD, incrementing values for r and R parameters.
# run GLD to minimize problem using all combinations of r={.001, .01, .1, 1} and R={10, 100, 1000, 10000}.
# generate a heat map with final function evaluations.

import pycutest
from Algorithms.gld_optimizer import GLDOptimizer
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from ExampleCode.oracle import Oracle, Oracle_pycutest
import copy

# problem 1: ROSENBR.
# problem 2: HILBERTA.
# problem 3: WATSON.

p1_name = 'ROSENBR'
problem_1 = pycutest.import_problem(p1_name)
p2_name = 'HILBERTA'
problem_2 = pycutest.import_problem(p2_name)
p3_name = 'WATSON'
problem_3 = pycutest.import_problem(p3_name)

# select which problem you want.
choose_problem = problem_3
choose_name = p3_name
# obtain initial value (same for each run of the same problem).
choose_x0 = choose_problem.x0

# parameters.
query_budget = 100
R = 10
r = 0.001
# lists.
list_R = []
list_r = []
list_evaluation = []

# ---------
# GLD - version of the code that averages across 5 runs.
for i in range(4):
    r = 0.001
    for j in range(4):
        p = choose_problem
        x0_gld = copy.copy(choose_x0)
        oracle_gld = Oracle_pycutest(p.obj)
        list_evals = []
        for k in range(5):
            gld1 = GLDOptimizer(oracle_gld, query_budget, x0_gld, R, r, p.obj)
            # step.
            termination = False
            prev_evals = 0
            while termination is False:
                solution, func_value, termination, queries = gld1.step()
            print('function evaluation at solution: ', func_value[-1])
            list_evals.append(func_value[-1])
        # take average of 5 evaluations.
        avg_eval = sum(list_evals) / len(list_evals)
        print('R: ', R)
        print('r: ', r)
        list_R.append(R)
        list_r.append(r)
        list_evaluation.append(avg_eval)
        # increment r.
        r *= 10
    # increment R.
    R *= 10

# checks.
print('\n')
print('R: ', list_R)
print('r: ', list_r)
print('last function value: ', list_evaluation)
list_symbol = [choose_name for i in range(len(list_evaluation))]
print('symbol: ', list_symbol)

# reshape function evaluations and problem name.
evaluations = np.array(list_evaluation).reshape(4, 4)
symbols = np.array(list_symbol).reshape(4, 4)
print('\n')
print('evaluations: ')
print(evaluations)
print('symbols: ')
print(symbols)

# create dataframe of symbols, eval, r, R.
df1 = pd.DataFrame(list(zip(list_symbol, list_evaluation, list_r, list_R)),
                   columns=['Symbols', 'Last_Function_Eval', 'r_Values', 'R_Values'])

# rows and columns of heatmap.
rows = np.unique(np.array(list_r))
columns = np.unique(np.array(list_R))
print('rows: ', rows)
print('columns: ', columns)
print('\n')

# use function pivot.
result = df1.pivot(index='R_Values', columns='r_Values', values='Last_Function_Eval')
print('result: ')
print(result)

# generate labels.
labels = (np.asarray(
    ["{0} \n {1:.2f}".format(symb, value) for symb, value in zip(symbols.flatten(), evaluations.flatten())])).reshape(4,
                                                                                                                      4)

# plot heatmap.
fig, ax = plt.subplots(figsize=(12, 7))
title = "GLD Algorithm - " + str(choose_name) + " Heat Map"
plt.title(title, fontsize=18)
tt1 = ax.title
tt1.set_position([0.5, 1.05])
sns.heatmap(result, annot=labels, fmt="", cmap='BuPu', linewidth=.3, ax=ax)
ax.set_xlabel('r', fontsize=12)
ax.set_ylabel('R', fontsize=12)
plt.ylabel('R')
plt.show()
plt.close()
# run this code for Problem_1, Problem_2, Problem_3.
