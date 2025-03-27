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

# Create ICE Plot for the feature "MedInc" (Median Income)
fig, ax = plt.subplots(figsize=(10, 6))
display = PartialDependenceDisplay.from_estimator(
    model, X, features=[0], kind="individual", ax=ax, feature_names=data.feature_names
)
ax.set_title("ICE Plot for MedInc (Median Income)")
ax.set_xlabel("MedInc (Median Income)")
ax.set_ylabel("Predicted House Price")
plt.show()
