import numpy as np
import pandas as pd

# Generate a binary dataset (-1 and 1) for 10 memories with 10 values each
np.random.seed(42)  # For reproducibility
data = np.random.choice([-1, 1], size=(10, 10))  # Randomly assign -1 or 1

# Create a DataFrame for clarity
df = pd.DataFrame(data, columns=[f"v{i}" for i in range(10)])

# Convert binary data to probability distributions
# We will treat each column as a distribution of -1 and 1
# So, we need to compute the probabilities for each value (-1 and 1) in each column
def get_probabilities(column):
    # Count occurrences of -1 and 1 in the column
    p_neg1 = np.sum(column == -1) / len(column)
    p_1 = 1 - p_neg1  # Since it's binary, p_1 = 1 - p_neg1
    return np.array([p_neg1, p_1])

# Function to compute KL Divergence between two probability distributions
def kl_divergence(P, Q):
    # Ensure the distributions are valid and avoid log(0) by adding a small epsilon to Q
    epsilon = 1e-10
    Q = np.clip(Q, epsilon, 1)
    return np.sum(P * np.log(P / Q))

# Function to compute Total Variation (TV) Distance between two probability distributions
def tv_distance(P, Q):
    return 0.5 * np.sum(np.abs(P - Q))

# Initialize the KL and TV distance matrices
kl_matrix = np.zeros((10, 10))
tv_matrix = np.zeros((10, 10))

# Calculate the KL Divergence and TV Distance for each pair of columns
for i in range(10):
    for j in range(i + 1, 10):  # Only calculate for pairs (i, j) where i < j to avoid repetition
        # Get the probability distributions for both columns
        P = get_probabilities(df.iloc[:, i])
        Q = get_probabilities(df.iloc[:, j])
        
        # Calculate the KL Divergence and TV Distance for the pair
        kl_value = kl_divergence(P, Q)
        tv_value = tv_distance(P, Q)
        
        # Store the values in the matrices (symmetry)
        kl_matrix[i, j] = kl_matrix[j, i] = kl_value
        tv_matrix[i, j] = tv_matrix[j, i] = tv_value

# Convert to DataFrames for clarity
kl_df = pd.DataFrame(kl_matrix, columns=df.columns, index=df.columns)
tv_df = pd.DataFrame(tv_matrix, columns=df.columns, index=df.columns)

# Visualize the KL Divergence Matrix (Heatmap)
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 8))
sns.heatmap(kl_df, annot=True, cmap="Blues", center=0)
plt.title("KL Divergence Heatmap")
plt.show()

# Visualize the TV Distance Matrix (Heatmap)
plt.figure(figsize=(10, 8))
sns.heatmap(tv_df, annot=True, cmap="Reds", center=0)
plt.title("Total Variation Distance Heatmap")
plt.show()

