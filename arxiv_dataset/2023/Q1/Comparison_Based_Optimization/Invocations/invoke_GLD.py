# invokes the GLD algorithm class.

from Algorithms.gld_optimizer import GLDOptimizer
from ExampleCode.benchmark_functions import SparseQuadratic, MaxK
from matplotlib import pyplot as plt
import numpy as np
from ExampleCode.oracle import Oracle

# ---------
print('sample invoke.')
# GLD - FUNCTION sample invocation.
n_def = 200  # problem dimension.
s_exact = 20  # True sparsity.
noise_amp = 0.001  # noise amplitude.
# initialize objective function.
obj_func_1 = SparseQuadratic(n_def, s_exact, noise_amp)
obj_func_2 = MaxK(n_def, s_exact, noise_amp)
query_budget = int(1e3)
x_0_ = np.random.rand(n_def)
print('shape of x_0_: ', len(x_0_))
R_ = 10
r_ = 1e-3

# Define the comparison oracle.
oracle = Oracle(obj_func_1)

# GLDOptimizer instance.
gld1 = GLDOptimizer(oracle, query_budget, x_0_, R_, r_, function=obj_func_1)
# step.
termination = False
prev_evals = 0
while termination is False:
    # optimization step.
    solution, func_value, termination, queries = gld1.step()
    print('current value: ', func_value[-1])

print('\n')
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
print('\n')
print('number of function vals: ', len(func_value))
print('function evaluations: ', func_value)
