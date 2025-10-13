import torch
from torch import nn
import torch.nn.functional as F
from abc import abstractmethod, ABC

class SparseAutoencoder(nn.Module, ABC):
    """Abstract class for Sparse Autoencoders"""
    def __init__(self, d_vit = 768, expansion_factor = 16, **kwargs):
        """Initialize W_enc = W_dec.T, according to 
        "Scaling and evaluating sparse autoencoders" (Gao et al.)"""
        super().__init__()

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        d_sae = d_vit * expansion_factor
        
        self.W_enc = nn.Parameter(
            torch.nn.init.kaiming_uniform_(torch.empty(d_vit, d_sae))
        )

        self.b_dec = nn.Parameter(torch.zeros(d_vit))
        self.W_dec = nn.Parameter(
            torch.nn.init.kaiming_uniform_(torch.empty(d_sae, d_vit))
        )

        self.W_dec.data[:] = self.W_enc.t().data # Initialize W_dec = W_enc.T
        self.W_dec.data[:] = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)

    @abstractmethod
    def forward(self, x):
        """
        Compute reconstruction of input x
        Parameters:
            - x (tensor): batch of vit activations
        Return:
            - x_hat (tensor) : reconstruction of input x
        """
        pass

    @abstractmethod
    def get_cls_features(self, x):
        """Get SAE features for a batch of activations, CLS token"""
        pass

    @abstractmethod
    def loss(self, x, x_hat):
        """Get loss of SAE. Objective is to be flexible between MSE, MSE+L1..."""
        pass


class TopKSAE(SparseAutoencoder):
    """SAE constraining directly L0 over features, per patch"""
    def __init__(self, d_vit=768, expansion_factor=16, top_k=32,**kwargs):
        super().__init__(d_vit, expansion_factor, **kwargs)
        self.top_k = top_k

    def forward(self, x):
        x_cent = x - self.b_dec
        acts = F.relu(x_cent @ self.W_enc)
        features = torch.topk(acts, self.top_k, dim=-1)
        features = torch.zeros_like(acts).scatter(
            -1, features.indices, features.values
        )

        x_hat = features @ self.W_dec + self.b_dec
        return x_hat
    

    def get_cls_features(self, x):
        x_cent = x - self.b_dec
        acts = F.relu(x_cent @ self.W_enc)

        features = acts
        return features[...,0,:]
    
    def loss(self, x, x_hat):
        """Simple MSE loss for TopKSAE"""

        mse_loss = nn.MSELoss().to(self.device)(x, x_hat)

        return mse_loss