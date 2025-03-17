import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances_argmin_min

# Load the Iris dataset
data = load_iris()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train a k-NN classifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_scaled, y)

# Function to find nearest neighbor counterfactual
def find_counterfactual(instance, model, X, y):
    original_class = model.predict([instance])[0]
    # Find the nearest neighbor from a different class
    mask = y != original_class
    candidates = X[mask]
    indices, distances = pairwise_distances_argmin_min([instance], candidates)
    counterfactual = candidates[indices[0]]
    return counterfactual

# Select a sample instance and find its counterfactual
sample_index = 0
sample_instance = X_scaled[sample_index]
counterfactual = find_counterfactual(sample_instance, knn, X_scaled, y)

print("Original instance:", sample_instance)
print("Counterfactual instance:", counterfactual)