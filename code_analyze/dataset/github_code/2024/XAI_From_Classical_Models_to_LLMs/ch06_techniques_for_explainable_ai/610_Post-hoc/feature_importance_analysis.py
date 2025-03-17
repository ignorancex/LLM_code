from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

# Load the Iris dataset
data = load_iris()
X, y = data.data, data.target

# Train a Decision Tree Classifier
model = DecisionTreeClassifier(random_state=42)
model.fit(X, y)

# Extract feature importance
importances = model.feature_importances_
feature_names = data.feature_names

# Plot feature importance
plt.figure(figsize=(8, 6))
plt.barh(feature_names, importances, color='skyblue')
plt.xlabel("Feature Importance Score")
plt.ylabel("Features")
plt.title("Feature Importance Analysis for Decision Tree Classifier")
plt.show()