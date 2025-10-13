import numpy as np
from scaling import calc_c

xor = np.array([[-1, -1, -1], [-1, 1, 1], [1, -1, 1], [1, 1, -1]])
rows = xor.shape[0]
cols = xor.shape[1]

"""
print(calc_c(rows,cols,2))
print(calc_c(rows,cols,3))
print(calc_c(rows,cols,4))
print(calc_c(rows,cols,5))
"""

print(
    calc_c(50, 16, 4)
)  # rougly gives an indication of error rate, but not between [0,1]
# meh sorta working
