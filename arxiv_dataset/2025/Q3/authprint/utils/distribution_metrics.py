"""
Distribution-level metrics for comparing image distributions.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import models
from scipy.stats import entropy
from scipy.spatial.distance import cdist


class InceptionScore(nn.Module):
    """Inception Score calculator using pretrained InceptionV3."""
    def __init__(self, device='cpu'):
        super().__init__()
        self.model = models.inception_v3(pretrained=True, transform_input=False).to(device)
        self.model.fc = nn.Identity()  # Remove final FC layer
        self.model.eval()
        
    @torch.no_grad()
    def calculate_score(self, images, batch_size=50, splits=10):
        """Calculate Inception Score.
        
        Args:
            images: Tensor of images in range [0, 1]
            batch_size: Batch size for processing
            splits: Number of splits for computing mean/std
            
        Returns:
            tuple: (mean_score, std_score)
        """
        self.model.eval()
        preds = []
        
        # Get predictions
        for i in range(0, images.size(0), batch_size):
            batch = images[i:i+batch_size]
            if batch.size(-1) != 299:
                batch = F.interpolate(batch, size=(299, 299), mode='bilinear', align_corners=False)
            pred = F.softmax(self.model(batch), dim=1)
            preds.append(pred.cpu().numpy())
        
        preds = np.concatenate(preds, axis=0)
        
        # Split predictions and calculate scores
        scores = []
        for k in range(splits):
            part = preds[k * (len(preds) // splits): (k + 1) * (len(preds) // splits), :]
            kl = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
            kl = np.mean(np.sum(kl, 1))
            scores.append(np.exp(kl))
        
        return np.mean(scores), np.std(scores)


def calculate_kid(features1, features2, subset_size=1000):
    """Calculate Kernel Inception Distance.
    
    Args:
        features1: Features from first distribution
        features2: Features from second distribution
        subset_size: Size of subset to use for estimation
        
    Returns:
        float: KID score
    """
    # Select random subsets
    m = min(features1.shape[0], features2.shape[0], subset_size)
    idx1 = np.random.choice(features1.shape[0], m, replace=False)
    idx2 = np.random.choice(features2.shape[0], m, replace=False)
    
    x = features1[idx1]
    y = features2[idx2]
    
    # Kernel MMD
    xx = np.dot(x, x.T)
    yy = np.dot(y, y.T)
    xy = np.dot(x, y.T)
    
    return np.mean(xx) + np.mean(yy) - 2 * np.mean(xy)


def calculate_precision_recall(real_features, fake_features, k=3, num_samples=10000):
    """Calculate precision and recall scores for distributions.
    
    Args:
        real_features: Features from real distribution
        fake_features: Features from fake distribution
        k: Number of nearest neighbors
        num_samples: Number of samples to use
        
    Returns:
        tuple: (precision, recall)
    """
    def manifold_estimate(features, neighbor_features, k):
        # Compute pairwise distances
        distances = cdist(features, neighbor_features, metric='euclidean')
        radii = np.sort(distances, axis=1)[:, k]
        return radii
    
    # Subsample if needed
    if len(real_features) > num_samples:
        real_idx = np.random.choice(len(real_features), num_samples, replace=False)
        real_features = real_features[real_idx]
    if len(fake_features) > num_samples:
        fake_idx = np.random.choice(len(fake_features), num_samples, replace=False)
        fake_features = fake_features[fake_idx]
    
    real_manifold = manifold_estimate(real_features, real_features, k)
    fake_manifold = manifold_estimate(fake_features, fake_features, k)
    
    precision = np.mean(manifold_estimate(fake_features, real_features, k-1) <= real_manifold)
    recall = np.mean(manifold_estimate(real_features, fake_features, k-1) <= fake_manifold)
    
    return precision, recall


def calculate_wasserstein(features1, features2, num_projections=1000):
    """Calculate approximate Wasserstein distance using random projections.
    
    Args:
        features1: Features from first distribution
        features2: Features from second distribution
        num_projections: Number of random projections
        
    Returns:
        float: Approximate Wasserstein distance
    """
    dim = features1.shape[1]
    
    # Generate random projections
    projections = np.random.normal(size=(num_projections, dim))
    projections = projections / np.sqrt(np.sum(projections ** 2, axis=1, keepdims=True))
    
    # Project features
    proj1 = np.dot(features1, projections.T)
    proj2 = np.dot(features2, projections.T)
    
    # Sort projections
    proj1 = np.sort(proj1, axis=0)
    proj2 = np.sort(proj2, axis=0)
    
    # Calculate Wasserstein-1
    return np.mean(np.abs(proj1 - proj2))


def calculate_mmd(features1, features2, sigma=1.0):
    """Calculate Maximum Mean Discrepancy with Gaussian kernel.
    
    Args:
        features1: Features from first distribution (numpy array)
        features2: Features from second distribution (numpy array)
        sigma: Kernel bandwidth
        
    Returns:
        float: MMD score
    """
    def gaussian_kernel(x, y, sigma):
        x = torch.from_numpy(x)
        y = torch.from_numpy(y)
        dist = torch.sum((x.unsqueeze(1) - y.unsqueeze(0)) ** 2, dim=-1)
        return torch.exp(-dist / (2 * sigma ** 2))
    
    xx = gaussian_kernel(features1, features1, sigma)
    yy = gaussian_kernel(features2, features2, sigma)
    xy = gaussian_kernel(features1, features2, sigma)
    
    return float(torch.mean(xx) + torch.mean(yy) - 2 * torch.mean(xy)) 