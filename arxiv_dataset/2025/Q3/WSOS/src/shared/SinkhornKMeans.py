import numpy as np
from scipy.spatial.distance import cdist
from sklearn.base import BaseEstimator, ClusterMixin
import ot
import torch

class SinkhornKMeans(BaseEstimator, ClusterMixin):
    def __init__(self, max_iter, n_clusters=3, epsilon=1.0, tol=1e-4, device= None):
        self.n_clusters = n_clusters
        self.cluster_centers_ = None
        self.labels_ = None
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.tol = tol
        self.device = device

    def fit(self, X, device=None):
        """
        Fit the SinkhornKMeans clustering model to the data.

        Parameters:
            X: Input data (n_samples x n_features)

        Returns:
            self: Fitted clustering model
        """
        n_samples, n_features = X.shape
        # Initialize cluster centers randomly
        self.cluster_centers_ = X[np.random.choice(n_samples, self.n_clusters, replace=False)]

        for iteration in range(self.max_iter):
            # Compute cost matrix between samples and current cluster centers
            C = cdist(X, self.cluster_centers_, metric='sqeuclidean')

            b_constant = (X.shape[0]/self.n_clusters)
            if self.device is not None:
                C = torch.from_numpy(C).to(self.device)
                a = torch.ones(X.shape[0]).to(self.device)
                b = b_constant*torch.ones(self.n_clusters).to(self.device)
                method = 'sinkhorn_log'
            else:
                a = np.ones(X.shape[0])
                b = b_constant*np.ones(self.n_clusters)
                method = 'sinkhorn_epsilon_scaling'
            # Compute the optimal transport matrix
            Pi = ot.sinkhorn(a,b, C, self.epsilon, method=method, numItermax=1000).cpu().numpy()

            # Assign labels based on the highest transport probability for each sample
            new_labels = np.argmax(Pi, axis=1)

            # Update cluster centers based on transport matrix
            new_centers = np.array([
                np.sum(Pi[:, j][:, None] * X, axis=0) / np.sum(Pi[:, j])
                for j in range(self.n_clusters)
            ])

            # Check for convergence
            center_shift = np.linalg.norm(self.cluster_centers_ - new_centers, axis=None)
            self.cluster_centers_ = new_centers
            self.labels_ = new_labels
            if center_shift < self.tol:
                break

        return self

    def predict(self, X):
        """
        Predict the closest cluster each sample in X belongs to.

        Parameters:
            X: Input data (n_samples x n_features)

        Returns:
            labels: Index of the cluster each sample belongs to (n_samples,)
        """
        # Compute cost matrix to existing cluster centers and assign to closest cluster
        C = cdist(X, self.cluster_centers_, metric='sqeuclidean')
        # Pi = self._sinkhorn(C)
        labels = np.argmin(C, axis=1)
        return labels
    