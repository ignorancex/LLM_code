import numpy as np
import torch
import torch.nn as nn
import pdb

from typing import Optional
from functools import partial
from scipy import stats
from sklearn.metrics import pairwise_distances

from source.dataset_utils import collate_features
from .strategy import Strategy

# BADGE: Batch Active Learning by Diverse Gradient Embeddings (BADGE)

# The init_centers() function implements the k-means++ initialization algorithm to select initial cluster
# centers for k-means clustering. It takes as input an unlabeled dataset X and the number of clusters K to
# be created. The function initializes the first cluster center as the data point with the largest Euclidean norm,
# and then iteratively selects the remaining centers based on a probability distribution that favors data points that are
# far away from the already selected centers.

# The query() method of the BadgeSampling() class selects a subset of n unlabeled data points that are
# most informative for the machine learning model to be labeled by a human expert.
# It does so by computing the model's prediction probabilities and embeddings for the unlabeled data 
# points, and then selecting n data points that are farthest away from the already selected data
# points using the init_centers() function. 
# The method returns the selected data points' indices, embeddings, prediction probabilities, and the
# indices of the selected data points in the embedding space.

# kmeans ++ initialization
def init_centers(X, K):
    ind = np.argmax([np.linalg.norm(s, 2) for s in X])
    mu = [X[ind]]
    indsAll = [ind]
    centInds = [0.] * len(X)
    cent = 0
    
    while len(mu) < K:
        if len(mu) == 1:
            D2 = pairwise_distances(X, mu).ravel().astype(float) # ravel(): flattent a multi-dimensional numpy array into a 1d-array
        else:
            newD = pairwise_distances(X, [mu[-1]]).ravel().astype(float)
            for i in range(len(X)):
                if D2[i] > newD[i]:
                    centInds[i] = cent
                    D2[i] = newD[i]
        
        if sum(D2) == 0.0: pdb.set_trace()
        D2 = D2.ravel().astype(float)
        Ddist = (D2 ** 2) / sum(D2 ** 2)
        customDict = stats.rv_discrete(name='custm', values=(np.arange(len(D2)), Ddist)) # creates a random variable for a custom discrete probability distribution, which can then be used to generate random samples from that distribution.
        ind = customDict.rvs(size=1)[0]
        mu.append(X[ind])
        indsAll.append(ind)
        cent += 1
    gram = np.matmul(X[indsAll], X[indsAll].T)
    val, _ = np.linalg.eig(gram)
    val = np.abs(val)
    vgt = val[val > 1e-2]
    return indsAll

class BadgeSampling(Strategy):
    def __init__(
        self,
        model: nn.Module,
        pool_dataset: torch.utils.data.Dataset,
        train_dataset: torch.utils.data.Dataset,
        collate_fn: callable = partial(collate_features, label_type="int"),
        batch_size: Optional[int] = 1,
    ):
        super(BadgeSampling, self).__init__(model, pool_dataset, train_dataset, collate_fn, batch_size)
    
    def query(self, WSI_budget):
        probs, embedings = self.predict_prob_embed(self.model, self.pool_dataset, self.collate_fn, self.batch_size)
        _, idxs = probs.sort(descending=True)

        gradEmbedding = self.get_grad_embedding(self.model, self.pool_dataset, self.collate_fn, self.batch_size)
        gradEmbedding = gradEmbedding.numpy()
        selected = init_centers(gradEmbedding, WSI_budget)
        return selected
