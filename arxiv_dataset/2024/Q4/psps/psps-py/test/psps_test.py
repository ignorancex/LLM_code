import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import glm
from psps_py import psps

# Load data
lab = pd.read_csv("./psps/test_data/lab.csv")
unlab = pd.read_csv("./psps/test_data/unlab.csv")

# Fit logistic regression models and extract the second coefficient
est_lab_y = glm('Y ~ X', data=lab, family=sm.families.Binomial()).fit().params['X']
est_lab_yhat = glm('Yhat ~ X', data=lab, family=sm.families.Binomial()).fit().params['X']
est_unlab_yhat = glm('Yhat ~ X', data=unlab, family=sm.families.Binomial()).fit().params['X']

# Bootstrap for covariance calculation
B = 1000  # Number of bootstrap iterations
n = len(lab)  # Total observations in labeled data
np.random.seed(123)  # For reproducibility

est_lab_y_boot = np.zeros(B)
est_lab_yhat_boot = np.zeros(B)

for i in range(B):
    boot_indices = np.random.choice(lab.index, size=n, replace=True)
    boot_lab = lab.loc[boot_indices]
    est_lab_y_boot[i] = glm('Y ~ X', data=boot_lab, family=sm.families.Binomial()).fit().params['X']
    est_lab_yhat_boot[i] = glm('Yhat ~ X', data=boot_lab, family=sm.families.Binomial()).fit().params['X']

# Compute the covariance matrix
Sigma = np.zeros((3, 3))
Sigma[0:2, 0:2] = np.cov(np.vstack([est_lab_y_boot, est_lab_yhat_boot]))
Sigma[2, 2] = glm('Yhat ~ X', data=unlab, family=sm.families.Binomial()).fit().bse['X']**2

# Run psps
fit_psps = psps(est_lab_y, est_lab_yhat, est_unlab_yhat, Sigma)
print(fit_psps)
