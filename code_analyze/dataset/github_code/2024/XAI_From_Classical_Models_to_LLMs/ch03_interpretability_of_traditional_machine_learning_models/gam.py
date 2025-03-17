import numpy as np
import matplotlib.pyplot as plt
from pygam import LinearGAM, s

# Generate synthetic data
np.random.seed(42)
x1 = np.random.uniform(-3, 3, 200)
x2 = np.random.uniform(-3, 3, 200)
y = np.sin(x1) + 0.5 * np.cos(x2) + np.random.normal(0, 0.2, 200)

# Combine features into a matrix
X = np.column_stack((x1, x2))

# Define and fit the GAM model
gam = LinearGAM(s(0) + s(1))
gam.fit(X, y)

# Plot the partial dependence for each feature
fig, axs = plt.subplots(1, 2, figsize=(12, 5))
for i, ax in enumerate(axs):
    XX = gam.generate_X_grid(term=i)
    ax.plot(XX[:, i], gam.partial_dependence(term=i, X=XX))
    ax.set_title(f'Partial Dependence of Feature x{i+1}')
    ax.set_xlabel(f'x{i+1}')
    ax.set_ylabel('Predicted y')

plt.tight_layout()
plt.show()