import torch
import torch.nn.functional as F
from torch_geometric.datasets import KarateClub
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch_geometric.data import Data
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from torch_cluster import random_walk
from sklearn.linear_model import LogisticRegression
import networkx as nx
import seaborn as sns
import random
import pylab as py
from sklearn.metrics import f1_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
import scipy.sparse as sp
import warnings
warnings.filterwarnings("ignore")

EPS = 1e-15

class DeepWalk_Linear(torch.nn.Module):
    def __init__(self, num_nodes, embedding_dim, walk_length, context_size, probability_norm_tensor,
                 walks_per_node = 1, p = 1, q = 1, num_negative_samples=None):
        super(DeepWalk_Linear, self).__init__()
        assert walk_length >= context_size
        self.num_nodes = num_nodes
        self.embedding_dim = embedding_dim
        self.walk_length = walk_length - 1
        self.context_size = context_size
        self.walks_per_node = walks_per_node
        self.p = p
        self.q = q
        self.num_negative_samples = num_negative_samples
        self.probability_norm_tensor = probability_norm_tensor
        self.embedding = torch.nn.Embedding(num_nodes, embedding_dim)
        
        self.reset_parameters()
        
    def reset_parameters(self):
        """ Resets the embeddings """
        self.embedding.reset_parameters()
        
    def forward(self, subset):
        """ Returns the embeddings for the nodes in subset"""
        return self.embedding(subset)
    
    def __random_walk__(self, edge_index, subset = None):
        
        if subset is None:
            subset = torch.arange(self.num_nodes, device = edge_index.device)
        subset = subset.repeat(self.walks_per_node)
        
        rw = random_walk(edge_index[0], edge_index[1], subset,
                        self.walk_length, self.p, self.q, self.num_nodes)
        
        walks = []
        num_walks_per_rw = 1 + self.walk_length + 1 - self.context_size
        
        for j in range(num_walks_per_rw):
            walks.append(rw[:, j:j + self.context_size])
        return torch.cat(walks, dim=0)
    
    def loss(self, edge_index, subset=None):
        
        walk = self.__random_walk__(edge_index, subset)
        start, rest = walk[:, 0], walk[:, 1:].contiguous()
        
        h_start = self.embedding(start).view(
                walk.size(0), 1, self.embedding_dim)
        
        h_rest = self.embedding(rest.view(-1)).view(
                walk.size(0), rest.size(1), self.embedding_dim)
        
        out = (h_start * h_rest).sum(dim=-1).view(-1)
        pos_loss = -torch.log(torch.sigmoid(out) + EPS).mean()
        
        # Negative sampling loss.
        num_negative_samples = self.num_negative_samples
        if num_negative_samples is None:
            num_negative_samples = rest.size(1)
        
        #### code block for selecting neg_sample from probability_norm

        negative_samples_tensor = torch.multinomial(self.probability_norm_tensor, num_negative_samples, replacement=True).to(device)
        
        neg_sample = negative_samples_tensor[start]

        h_neg_rest = self.embedding(neg_sample)
        out = (h_start * h_neg_rest).sum(dim=-1).view(-1)
        neg_loss = -torch.log(1 - torch.sigmoid(out) + EPS).mean()
        
        del negative_samples_tensor
        del neg_sample
        
        return pos_loss + neg_loss
    
    def test(self, train_z, train_y, test_z, test_y, solver='lbfgs',
             multi_class='auto', *args, **kwargs):
        r"""Evaluates latent space quality via a logistic regression downstream
        task."""
        clf = LogisticRegression(solver=solver, multi_class=multi_class, *args,
                                 **kwargs).fit(train_z.detach().cpu().numpy(),
                                               train_y.detach().cpu().numpy())
        pred_label, true_label = clf.predict(test_z.detach().cpu().numpy()), test_y.detach().cpu().numpy()
        return f1_score(true_label, pred_label, average='macro')
    
    def test_predict(self, train_z, train_y, test_z, test_y, solver='lbfgs',
             multi_class='auto', *args, **kwargs):
        r"""Evaluates latent space quality via a logistic regression downstream
        task."""
        clf = LogisticRegression(solver=solver, multi_class=multi_class, *args,
                                 **kwargs).fit(train_z.detach().cpu().numpy(),
                                               train_y.detach().cpu().numpy())
        return clf.predict(test_z.detach().cpu().numpy()), test_y.detach().cpu().numpy()
    
    
    def test_predict_f1(self, train_z, train_y, test_z, test_y, solver='lbfgs',
             multi_class='auto', *args, **kwargs):
        r"""Evaluates latent space quality via a logistic regression downstream
        task."""
        pred_label, true_label = self.test_predict(train_z, train_y, test_z, test_y)
        f1_macro = f1_score(true_label, pred_label, average='macro')
        f1_micro = f1_score(true_label, pred_label, average='micro')
        return f1_macro, f1_micro

    def __repr__(self):
        return '{}({}, {}, p={}, q={})'.format(
            self.__class__.__name__, self.num_nodes, self.embedding_dim,
            self.p, self.q)
    