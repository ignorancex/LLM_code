from sklearn.metrics import jaccard_score
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Generate a binary dataset (-1 and 1) for 10 memories with 10 values each
np.random.seed(42)  # For reproducibility
data = np.random.choice([-1, 1], size=(10, 10))  # Randomly assign -1 or 1

# Convert -1 to 0 for Jaccard similarity computation
data_binary = (data == 1).astype(int)  # Convert -1 to 0, 1 stays 1

# Compute pairwise Jaccard similarity
n = data_binary.shape[0]
jaccard_sim_matrix = np.zeros((n, n))

for i in range(n):
    for j in range(n):
        if i != j:
            jaccard_sim_matrix[i, j] = jaccard_score(data_binary[i], data_binary[j])

# Convert to DataFrame for better readability
jaccard_sim_df = pd.DataFrame(jaccard_sim_matrix, columns=[f"Memory {i}" for i in range(10)], index=[f"Memory {i}" for i in range(10)])

# Visualize the Jaccard Similarity Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(jaccard_sim_df, annot=True, cmap="coolwarm", center=0)
plt.title("Pairwise Jaccard Similarity Heatmap")
plt.show()
