import numpy as np
import pandas as pd
from causality.inference.search import ICAlgorithm
from causality.estimation.adjustments import AdjustForDirectCauses
from causality.estimation.nonparametric import CausalEffect

# Define the structural equations
np.random.seed(42)
n_samples = 1000
X = np.random.normal(size=n_samples)
U_M = np.random.normal(size=n_samples)
M = 0.5 * X + U_M
U_Y = np.random.normal(size=n_samples)
Y = 0.3 * M + 0.2 * X + U_Y

# Create a DataFrame
data = pd.DataFrame({'X': X, 'M': M, 'Y': Y})

# Estimate the causal effect of X on Y using SCM
adjuster = AdjustForDirectCauses()
causal_effect_estimator = CausalEffect()
effect = causal_effect_estimator.estimate(data, 'X', 'Y', adjust_for=['M'])

print(f"Estimated causal effect of X on Y: {effect:.4f}")