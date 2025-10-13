

import torch
import numpy as np
from torch_geometric.nn.models import DeepGraphInfomax, GCN
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from yellowbrick.cluster import KElbowVisualizer
from tqdm import tqdm
import datasets
device = "cuda" if torch.cuda.is_available() else "cpu"
hidden_dim = 64
max_k = 100

def readout(pos_z, x, edge_index):
    return torch.mean(pos_z, 0)

def corrupt(x, edge_index):
    idx = torch.randperm(x.shape[0])
    shuf_fts = x[idx, :]
    return shuf_fts, edge_index

dataset_name = "CiteSeer"
dataset = datasets.get_dataset(dataset_name)

encoder = GCN(in_channels=dataset.x.shape[1],hidden_channels=hidden_dim, num_layers=2, out_channels=None, dropout=0.5)
model = DeepGraphInfomax(hidden_dim, encoder, readout, corrupt).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr = 0.005)
model.train()
for epoch in tqdm(range(100)):
    pos_z, neg_z, summary = model.forward(dataset.x, dataset.edge_index)
    loss = model.loss(pos_z, neg_z, summary)
    loss.backward()
    optimizer.step()
model.eval()
embeddings, _, _ = model.forward(dataset.x, dataset.edge_index)
embeddings = embeddings.detach().cpu().numpy()
combined = np.concatenate((embeddings,dataset.x),axis=1)



clusterer = KMeans()
visualizer = KElbowVisualizer(clusterer, k=(2,100),timings=False)
visualizer.fit(embeddings)
visualizer.show(outpath="class_estimation_" + dataset_name + ".pdf")
visualizer.show()

"""
current best hyperparams:
Clusterer: Kmedoids, k-medoids++ init
hidden_dim=64
lr = 0.005
epochs = 100
dropout = 0.5
"""
