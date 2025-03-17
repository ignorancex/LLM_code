import numpy as np
import pandas as pd
import dowhy
from dowhy import CausalModel
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# Generate synthetic data
np.random.seed(0)
data_size = 1000
X = np.random.rand(data_size, 3)
gender = np.random.choice([0, 1], size=data_size)  # Sensitive attribute (0 = male, 1 = female)
income = X[:, 0] + 0.5 * gender + np.random.normal(size=data_size)  # Outcome influenced by gender
data = pd.DataFrame({'income': income, 'gender': gender, 'feature1': X[:, 1], 'feature2': X[:, 2]})

# Split data into training and testing sets
train_data, test_data = train_test_split(data, test_size=0.2, random_state=0)

# Train a decision tree model
model = DecisionTreeClassifier()
model.fit(train_data[['gender', 'feature1', 'feature2']], (train_data['income'] > 0.5).astype(int))

# Define causal model using DoWhy
causal_model = CausalModel(
    data=train_data,
    treatment='gender',
    outcome='income',
    common_causes=['feature1', 'feature2']
)

# Identify and estimate the Average Causal Effect (ACE)
identified_estimand = causal_model.identify_effect()
estimate = causal_model.estimate_effect(identified_estimand, method_name="backdoor.linear_regression")
print(f"Estimated Average Causal Effect (ACE): {estimate.value:.4f}")