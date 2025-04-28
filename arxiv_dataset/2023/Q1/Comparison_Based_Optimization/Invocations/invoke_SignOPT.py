# invokes the SignOPT algorithm class.

from Algorithms.signopt_optimizer import SignOPT
import numpy as np
from ExampleCode.oracle import Oracle
import matplotlib.pyplot as plt
from ExampleCode.benchmark_functions import SparseQuadratic, MaxK

# Defining the function.
n_def = 2000
s_exact = 200
noise_amp = 0.001
# objective functions.
obj_func_1 = SparseQuadratic(n_def, s_exact, noise_amp)
obj_func_2 = MaxK(n_def, s_exact, noise_amp)
# invoke.
query_budget = int(1e3)
m = 100
x0 = np.random.randn(n_def)
step_size = 0.2
r = 0.1

# Define the comparison oracle.
oracle = Oracle(obj_func_1)
Opt = SignOPT(oracle, query_budget, x0, m, step_size, r, debug=False,
              function=obj_func_1)
# step.
termination = False
i = 0
while termination is False:
    # optimization step.
    solution, func_value, termination, queries = Opt.step()
    # print('step')
    print('current value at ' + str(i) + ': ', func_value[-1])
print('solution: ', solution)
# plot the decreasing function.
plt.plot(func_value)
plt.title("plot")
plt.show()
plt.close()
# log x-axis.
plt.semilogy(func_value)
plt.title("log plot")
plt.show()
plt.close()
# ---------
print('number of values: ', len(func_value))
print('number of oracle queries: ', queries)
