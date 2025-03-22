import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# 1. Generate synthetic data with 3 features for binary classification
X, y = make_classification(
    n_samples=100,          # Number of samples
    n_features=3,           # Number of features
    n_informative=3,        # Number of informative features
    n_redundant=0,          # No redundant features
    n_classes=2,            # Binary classification
    random_state=42         # Random seed for reproducibility
)

# 2. Split the data into training and testing sets (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. Train a Decision Tree Classifier with a maximum depth of 4
clf = DecisionTreeClassifier(max_depth=4, random_state=42)
clf.fit(X_train, y_train)

# 4. Extract the feature importance scores from the trained classifier
feature_importance = clf.feature_importances_

# 5. Define feature names for the plot (e.g., Feature 1, Feature 2, Feature 3)
features = np.array(['Feature 1', 'Feature 2', 'Feature 3'])

# 6. Plot a horizontal bar chart to visualize feature importance
plt.barh(features, feature_importance)
plt.xlabel('Importance Score')                # Label for the x-axis
plt.title('Feature Importance in Decision Tree')  # Title of the plot
plt.show()                                   # Display the plot
