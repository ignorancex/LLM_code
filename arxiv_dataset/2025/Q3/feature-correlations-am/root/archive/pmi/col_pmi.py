import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mutual_info_score

# Generate binary dataset (-1 and 1) for 10 memories and 10 features
np.random.seed(42)
data = np.random.choice([-1, 1], size=(10, 10))  # Randomly assign -1 or 1
df = pd.DataFrame(data, columns=[f"v{i}" for i in range(10)])

# Helper function to compute PMI
def compute_pmi(x, y):
    joint_prob = pd.crosstab(x, y, normalize=True).values  # Joint probabilities
    x_prob = np.sum(joint_prob, axis=1, keepdims=True)  # Marginal probs for x
    y_prob = np.sum(joint_prob, axis=0, keepdims=True)  # Marginal probs for y
    pmi_matrix = np.log2(joint_prob / (x_prob @ y_prob))  # Compute PMI
    pmi_matrix[np.isinf(pmi_matrix)] = 0  # Handle log(0) as 0
    return pmi_matrix

# --- 2. Column-wise PMI (between neurons) ---
n_cols = df.shape[1]
col_pmi_matrix = np.zeros((n_cols, n_cols))  # Initialize matrix

for i in range(n_cols):
    for j in range(n_cols):
        if i != j:
            # Calculate PMI between columns
            col_pmi_matrix[i, j] = mutual_info_score(df.iloc[:, i], df.iloc[:, j])

# Convert PMI matrices to DataFrames for clarity
col_pmi_df = pd.DataFrame(col_pmi_matrix, index=df.columns, columns=df.columns)

# --- Plot Column-wise PMI Heatmap ---
plt.figure(figsize=(8, 6))
sns.heatmap(col_pmi_df, annot=True, cmap="coolwarm", center=0)
plt.title("PMI Between Columns (Neurons)")
plt.xlabel("Neuron Index")
plt.ylabel("Neuron Index")
plt.show()
