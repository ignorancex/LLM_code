import json
import torch
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt



# Load the embedding data
embedding_path = "/shared/eng/pj20/lc/DeepRetrieval/outputs/msmarco_query_embeddings.pt"
embeddings = torch.load(embedding_path)

# Load the query texts
queries = []
with open('data/MS-MARCO/queries.all.tsv', 'r') as f:
    for line in f:
        qid, query = line.strip().split('\t')
        queries.append(query)

# Convert embeddings to numpy array if it's a tensor
embeddings = embeddings.numpy() if isinstance(embeddings, torch.Tensor) else embeddings

# Dimensionality reduction using PCA
# pca = PCA(n_components=0.5)  # You can adjust the number of components as needed
# embeddings_pca = pca.fit_transform(embeddings)
embeddings_pca = embeddings

print(embeddings_pca.shape)

# Set the number of clusters
num_clusters = 25


# Agglomerative Hierarchical Clustering is not work for large data
# Apply Agglomerative Hierarchical Clustering on PCA-reduced embeddings
# euclidean and ward linkage is not efficient for large data
# agglo = AgglomerativeClustering(n_clusters=num_clusters, metric='cosine', linkage='average')
# agglo.fit(embeddings_pca)



# Apply KMeans Clustering on PCA-reduced embeddings
kmeans = KMeans(n_clusters=num_clusters, random_state=667)
kmeans.fit(embeddings_pca)

# Get the cluster labels
labels_kmeans = kmeans.labels_

# Create a dictionary where key is cluster id and value is a list of queries in that cluster
cluster_dict = {}

# Group the queries by cluster label
for i, label in enumerate(labels_kmeans):
    label = int(label)
    if label not in cluster_dict:
        cluster_dict[label] = []
    cluster_dict[label].append(queries[i])



# Save the clustered queries to a JSON file
with open('outputs/clustered_queries.json', 'w') as json_file:
    json.dump(cluster_dict, json_file)