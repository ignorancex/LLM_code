import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mutual_info_score

# Generate binary dataset (-1 and 1) for 10 memories and 10 features
np.random.seed(42)
data = np.random.choice([-1, 1], size=(10, 10))  # Randomly assign -1 or 1
df = pd.DataFrame(data, columns=[f"v{i}" for i in range(10)])

# --- Column-wise Mutual Information Calculation ---
n_cols = df.shape[1]
col_mi_matrix = np.zeros((n_cols, n_cols))  # Initialize matrix

for i in range(n_cols):
    for j in range(n_cols):
        if i != j:
            # Calculate Mutual Information between columns
            col_mi_matrix[i, j] = mutual_info_score(df.iloc[:, i], df.iloc[:, j])

# Convert Mutual Information matrix to DataFrame for visualization
col_mi_df = pd.DataFrame(col_mi_matrix, index=df.columns, columns=df.columns)

# --- Plot Column-wise Mutual Information Heatmap ---
plt.figure(figsize=(8, 6))
sns.heatmap(col_mi_df, annot=True, cmap="coolwarm", center=0)
plt.title("Mutual Information Between Columns (Neurons)")
plt.xlabel("Neuron Index")
plt.ylabel("Neuron Index")
plt.show()
