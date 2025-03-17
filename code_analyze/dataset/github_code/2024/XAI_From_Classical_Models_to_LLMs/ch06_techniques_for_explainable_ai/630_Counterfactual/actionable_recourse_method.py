import numpy as np
from scipy.optimize import minimize

# Define the prediction function of the model (simplified example)
def predict(model, x):
    # Remove reshaping to ensure x remains a 1D array
    return model.predict(x)

# Action cost function: Assign higher costs to harder-to-change features
def action_cost(x, x_prime, weights):
    return np.sum(weights * np.abs(x - x_prime))

# Objective function for actionable recourse
def objective_function(x_prime, x, model, target_class, lambda_param, weights):
    distance = np.linalg.norm(x - x_prime)
    action_cost_value = action_cost(x, x_prime, weights)
    prediction_loss = 0 if predict(model, x_prime) == target_class else 1
    return distance + lambda_param * action_cost_value + prediction_loss

# Generate actionable recourse
def generate_actionable_recourse(model, x, target_class, weights, lambda_param=0.1):
    # Initialize the counterfactual with the original input
    x_prime = np.copy(x)

    # Optimize to find actionable recourse
    result = minimize(
        objective_function,
        x_prime,
        args=(x, model, target_class, lambda_param, weights),
        method='L-BFGS-B'
    )

    return result.x

# Example usage
x = np.array([30, 50000, 0.4])  # Example input features: age, income, debt-to-income ratio
weights = np.array([0.0, 0.5, 1.0])  # Higher cost for changing age, lower for financial habits
target_class = 1  # Desired outcome: Loan approval

# Assume we have a pre-trained model (pseudo-model for illustration)
class SimpleModel:
    def predict(self, x):
        # x is expected to be a 1D array
        return int(x[1] > 40000 and x[2] < 0.5)  # Simplified decision rule

model = SimpleModel()

# Generate actionable recourse
counterfactual = generate_actionable_recourse(model, x, target_class, weights)

print("Original input:", x)
print("Actionable recourse:", counterfactual)
