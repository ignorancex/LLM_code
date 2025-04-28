# invokes the STP algorithm class.

from Algorithms.stp_optimizer import STPOptimizer
from ExampleCode.benchmark_functions import SparseQuadratic, MaxK
from matplotlib import pyplot as plt
import numpy as np
from ExampleCode.oracle import Oracle

n = 20000  # problem dimension.
s_exact = 200  # true sparsity.
noiseamp = 0.001  # noise amplitude.
# initialize objective function.

obj_func_1 = SparseQuadratic(n, s_exact, noiseamp)
obj_func_2 = MaxK(n, s_exact, noiseamp)
direction_vector_type = 3  # rademacher.
step_size = 0.1  # step-size.
query_budget = int(1e3)
m = 10
x0 = np.random.randn(n)

# Define the comparison oracle.
oracle = Oracle(obj_func_1)
target_func_value = 0

# stp instance.
stp = STPOptimizer(oracle, query_budget, x0, step_size, function=obj_func_1)

# step
termination = False
while termination is False:
    solution, func_value, termination, queries = stp.step()
    if func_value[-1] <= target_func_value:
        termination = True

print('f vals: ', stp.f_vals)
print('queries: ', stp.queries)
print('solution: ', solution)

# plot the decreasing function.
plt.plot(stp.f_vals)
plt.title("plot")
plt.show()
plt.close()
# log x-axis.
plt.semilogy(stp.f_vals)
plt.title("log plot")
plt.show()
plt.close()
# ---------
print('\n')
print('number of function vals: ', len(stp.f_vals))
