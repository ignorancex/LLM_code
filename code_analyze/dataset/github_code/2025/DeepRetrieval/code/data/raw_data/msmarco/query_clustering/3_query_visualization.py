# t-sne is too slow for large high-dimensional data
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Load the embeddings data
embedding_path = "/shared/eng/pj20/lc/DeepRetrieval/outputs/msmarco_query_embeddings.pt"
embeddings = torch.load(embedding_path)

# Ensure embeddings are of type NumPy array
embeddings = embeddings.numpy() if isinstance(embeddings, torch.Tensor) else embeddings

# Use t-SNE to reduce the embeddings to 2D
tsne = TSNE(n_components=2, random_state=42)
embeddings_2d = tsne.fit_transform(embeddings)

# Visualize the embeddings in 2D
plt.figure(figsize=(10, 8))
plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c='blue', marker='o', s=20)
plt.title("2D Visualization of Embeddings using t-SNE")
plt.xlabel("TSNE Component 1")
plt.ylabel("TSNE Component 2")
plt.grid(True)
plt.show()

# Save the plot as an image
plt.savefig("query_embedding_2d_visualization.png")