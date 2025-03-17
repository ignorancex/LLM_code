import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols

# Generate synthetic data
np.random.seed(42)
n = 100
X = np.random.randn(n)
M = 0.5 * X + np.random.randn(n) * 0.5  # Mediator influenced by X
Y = 0.3 * X + 0.7 * M + np.random.randn(n) * 0.5  # Outcome influenced by X and M

# Create a DataFrame
data = pd.DataFrame({'X': X, 'M': M, 'Y': Y})

# Step 1: Fit the mediator model (M ~ X)
mediator_model = ols('M ~ X', data=data).fit()

# Step 2: Fit the outcome model (Y ~ X + M)
outcome_model = ols('Y ~ X + M', data=data).fit()

# Extract coefficients
alpha_1 = mediator_model.params['X']
beta_1 = outcome_model.params['X']
beta_2 = outcome_model.params['M']

# Calculate direct, indirect, and total effects
indirect_effect = alpha_1 * beta_2
direct_effect = beta_1
total_effect = direct_effect + indirect_effect
proportion_mediated = indirect_effect / total_effect

print(f"Indirect Effect (IE): {indirect_effect:.4f}")
print(f"Direct Effect (DE): {direct_effect:.4f}")
print(f"Total Effect (TE): {total_effect:.4f}")
print(f"Proportion Mediated: {proportion_mediated:.2%}")