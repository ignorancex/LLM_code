"""
@author: hdsullivan
"""

import numpy as np
from ResSR.ressr import res_sr
import time

HR_MSI_SIDE_LENGTH = 4380

# Create random data y
y_10m = np.random.rand(HR_MSI_SIDE_LENGTH, HR_MSI_SIDE_LENGTH, 4)
y_20m = np.random.rand(HR_MSI_SIDE_LENGTH // 2, HR_MSI_SIDE_LENGTH // 2, 6)
y_60m = np.random.rand(HR_MSI_SIDE_LENGTH // 6, HR_MSI_SIDE_LENGTH // 6, 2)
y = [y_10m, y_20m, y_60m]

# Define necessary parameters
params = {}
params['sigma'] = 0.02
params['N_s'] = HR_MSI_SIDE_LENGTH
params['K'] = 2
params['gamma_hr'] = 0.99
params['lam'] = 0.5

# Run ResSR algorithm
tic = time.time()
x = res_sr(y, params['sigma'], params['N_s'], params['K'], params['gamma_hr'], params['lam'])
toc = time.time()

print("Time taken to run ResSR: ", toc - tic)