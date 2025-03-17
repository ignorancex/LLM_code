import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Define a sample attention weights matrix (3x3 for simplicity)
attention_weights = np.array([[0.1, 0.2, 0.7], 
                              [0.5, 0.3, 0.2], 
                              [0.3, 0.4, 0.3]])

# Create the heatmap plot
plt.figure(figsize=(6, 5))
sns.heatmap(attention_weights, annot=True, fmt=".2f", cmap='Blues', cbar=False, linewidths=0.5)

# Add titles and labels
plt.title("Sample Attention Heatmap", fontsize=14)
plt.xlabel("Input Tokens", fontsize=12)
plt.ylabel("Output Tokens", fontsize=12)
plt.xticks(ticks=[0.5, 1.5, 2.5], labels=["Token 1", "Token 2", "Token 3"])
plt.yticks(ticks=[0.5, 1.5, 2.5], labels=["Output 1", "Output 2", "Output 3"])

# Display the plot
plt.tight_layout()
plt.show()
