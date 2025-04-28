# invokes the SCOBO algorithm class.

from Algorithms.scobo_optimizer import SCOBOoptimizer
import numpy as np
from ExampleCode.oracle import Oracle
import matplotlib.pyplot as plt
from ExampleCode.benchmark_functions import SparseQuadratic, MaxK

# Defining the function.
n_def = 200
s_exact = 20
noise_amp = 0.001
func = SparseQuadratic(n_def, s_exact, noise_amp)
query_budget = int(1e3)
m = 100  # Should always be larger than s_exact.
x0 = np.random.randn(n_def)
step_size = 0.01
r = 0.1

# Define the comparison oracle.
oracle = Oracle(func)
Opt = SCOBOoptimizer(oracle, step_size, query_budget, x0, r, m, s_exact, function=func)  # NEW.
# step.
termination = False
prev_evals = 0
while termination is False:
    # optimization step.
    solution, func_value, termination, queries = Opt.step()
    print('current value at ' + str(prev_evals) + ': ', func_value[-1])
    prev_evals += 1
# print the solution.
print('\n')
print('solution: ', solution)
# plot the decreasing function.
plt.plot(func_value)
plt.title("plot")
plt.show()
plt.close()
plt.semilogy(func_value)
plt.title("log plot")
plt.show()
plt.close()
# ---------
print('number of values: ', len(func_value))
print('number of oracle queries: ', queries)
