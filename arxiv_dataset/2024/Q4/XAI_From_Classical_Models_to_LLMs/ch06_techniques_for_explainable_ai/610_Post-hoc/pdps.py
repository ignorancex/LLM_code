import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import fetch_california_housing
from sklearn.inspection import PartialDependenceDisplay

# Load the California Housing dataset
data = fetch_california_housing()
X, y = data.data, data.target

# Train a Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# Plot Partial Dependence for the feature "MedInc" (Median Income)
fig, ax = plt.subplots(figsize=(8, 6))
PartialDependenceDisplay.from_estimator(model, X, [0], feature_names=data.feature_names, ax=ax)
ax.set_title("Partial Dependence Plot for MedInc (Median Income)")
ax.set_xlabel("MedInc (Median Income)")
ax.set_ylabel("Predicted House Price")
plt.show()
