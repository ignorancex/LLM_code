# we want to figure out optimal values for s_exact & m in the SCOBO algorithm.

"""
experimenting with tuning SCOBO hyperparameters.
For PyCutest functionality, add this to ENVIRONMENT VARIABLES (edit configurations):
PYCUTEST_CACHE=.;CUTEST=/usr/local/opt/cutest/libexec;MYARCH=mac64.osx.gfo;SIFDECODE=/usr/local/opt/sifdecode/libexec;MASTSIF=/usr/local/opt/mastsif/share/mastsif;ARCHDEFS=/usr/local/opt/archdefs/libexec
"""

# INSTRUCTIONS:
# choose 3 PyCutest problems.
# run SCOBO, incrementing values for s_exact and m parameters.
# run SCOBO to minimize problem using all combinations of s_exact={10, 20, 50, 100} & m={100, 200, 500, 1000}.
# generate a heat map with final function evaluations.

import pycutest
from Algorithms.scobo_optimizer import SCOBOoptimizer
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

# select which problem you want. (EDIT choose_problem & choose_name.)
choose_problem = problem_3
choose_name = p3_name
# obtain initial value (same for each run of the same problem).
choose_x0 = choose_problem.x0

# parameters.
query_budget = 1e4
s_exact_initialize_list = [10, 20, 50, 100]
m_initialize_list = [100, 200, 500, 1000]
# lists.
list_s_exact = []
list_m = []
list_evaluation = []

# SCOBO - version of the code that averages across 5 runs:
for i in range(4):
    s_exact = s_exact_initialize_list[i]
    for j in range(4):
        m = m_initialize_list[j]
        p = choose_problem
        x0_scobo = copy.copy(choose_x0)
        step_size = 0.01
        r_scobo = 0.1
        oracle_scobo = Oracle_pycutest(p.obj)
        list_evals = []
        for k in range(5):
            scobo1 = SCOBOoptimizer(oracle_scobo, step_size, query_budget, x0_scobo, r_scobo, m, s_exact, p.obj)
            # step.
            termination = False
            prev_evals = 0
            while termination is False:
                # optimization step.
                solution, func_value, termination, queries = scobo1.step()
                prev_evals += 1
            print('function evaluation at solution: ', func_value[-1])
            list_evals.append(func_value[-1])
        # take average of 5 evaluations.
        avg_eval = sum(list_evals) / len(list_evals)
        print('s_exact: ', s_exact)
        print('m: ', m)
        list_s_exact.append(s_exact)
        list_m.append(m)
        list_evaluation.append(avg_eval)
        # m will be automatically incremented.
    # s_exact will be automatically incremented.

print('\n')
print('s_exact: ', list_s_exact)
print('m: ', list_m)
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
df1 = pd.DataFrame(list(zip(list_symbol, list_evaluation, list_m, list_s_exact)),
                   columns=['Symbols', 'Last_Function_Eval', 'm_Values', 's_exact_Values'])

# rows and columns of heatmap.
rows = np.unique(np.array(list_m))
columns = np.unique(np.array(list_s_exact))
print('rows: ', rows)
print('columns: ', columns)
print('\n')

# use function pivot.
result = df1.pivot(index='s_exact_Values', columns='m_Values', values='Last_Function_Eval')
print('result: ')
print(result)

# generate labels.
labels = (np.asarray(
    ["{0} \n {1:.2f}".format(symb, value) for symb, value in zip(symbols.flatten(), evaluations.flatten())])).reshape(4,
                                                                                                                      4)

# plot heatmap.
fig, ax = plt.subplots(figsize=(12, 7))
title = "SCOBO Algorithm - " + str(choose_name) + " Heat Map"
plt.title(title, fontsize=18)
tt1 = ax.title
tt1.set_position([0.5, 1.05])
sns.heatmap(result, annot=labels, fmt="", cmap='rocket', linewidth=.3, ax=ax)
ax.set_xlabel('m', fontsize=12)
ax.set_ylabel('s_exact', fontsize=12)
plt.show()
plt.close()
# run this code for Problem_1, Problem_2, Problem_3.
