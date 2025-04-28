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
query_budget = 10000
m = 10
x0 = np.random.randn(n)

# Define the comparison oracle.
oracle = Oracle(obj_func_1)

# stp instance.
stp = STPOptimizer(oracle, query_budget, x0, m, step_size,
                    direction_vector_type, function=obj_func_1)
# step.
termination = False
prev_evals = 0
while termination is False:
    # optimization step.
    solution, func_value, termination, queries = stp.step()
    # print('step')
    print('current value: ', func_value[-1])
# plot the decreasing function.
plt.plot(func_value)
plt.show()
plt.close()
# log x-axis.
plt.semilogy(func_value)
plt.show()
plt.close()
# ---------
print('\n')
print('number of function vals: ', len(func_value))


