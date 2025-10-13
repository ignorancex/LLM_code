import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Generate a binary dataset (-1 and 1) for 10 memories with 10 values each
np.random.seed(42)  # For reproducibility
data = np.random.choice([-1, 1], size=(10, 10))  # Randomly assign -1 or 1

# Create a DataFrame for clarity
df = pd.DataFrame(data, columns=[f"v{i}" for i in range(10)])

# Compute pairwise cosine similarity
cosine_sim_matrix = cosine_similarity(df)

# Convert to DataFrame for better readability
cosine_sim_df = pd.DataFrame(cosine_sim_matrix, columns=df.index, index=df.index)

# Visualize the Cosine Similarity Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(cosine_sim_df, annot=True, cmap="coolwarm", center=0)
plt.title("Pairwise Cosine Similarity Heatmap")
plt.show()
