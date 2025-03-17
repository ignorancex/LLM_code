# Import libraries
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
    
class TClassifier(nn.Module):
    """
    Custom Classification Transformer model with:
        - encoder layer
        - temperature-(single-head)-self-attention with normalization included
        - decoder layer
    """
    def __init__(self, vocab_size, emb=16, n=128, depth=4, temperature = 1):
        super(TClassifier, self).__init__()
        self.depth = depth
        self.encoder = nn.Embedding(vocab_size, emb)
        self.attention = CustomSelfAttention(emb, temperature)
        self.decoder = nn.Linear(emb, 1) #output_dim = 1 to classify
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x = x.long() #long tensor formatting (integers)
        x = self.encoder(x)
        for _ in range(self.depth): # share weights across layers
             x = self.attention(x)
        x = x.mean(dim=1)  # Taking the mean over all embedded words
        x = self.decoder(x)  # Project back to \R for binary classification
        x = self.sigmoid(x) # Normalize values to [0, 1] for binary classification
        return x
    
class CustomSelfAttention(nn.Module):
    """
    Custom implementation of (single)-head self-attention with:
     - temperature
     - K = Q = Id, and V = alpha * Id
     - normalization by 1 / ( 1 + alpha)
    """
    def __init__(self, emb, temperature):
        super().__init__()
        self.alpha = nn.Parameter(torch.Tensor(1))
        nn.init.uniform_(self.alpha)  # Initialize alpha with random values (uniform distribution U(0,1))
        self.emb = emb
        self.temperature = temperature
    def forward(self, x):
        # Rescaling factor
        rescale = 1 / (1 + self.alpha) 
        # Get dot product of queries (x) and keys (x)
        dot = torch.bmm(x, x.transpose(-2, -1))
        # Divide dot by the temperature parameter and apply row-wise softmax
        dot = dot / self.temperature
        dot = F.softmax(dot, dim=2)
        # apply similarity matrix to the values (\alpha * x)
        out = x + torch.bmm(dot, self.alpha * x)
        # rescale by 1 / (1 + \alpha) 
        out = rescale * out
        return out

def encoder(E, features):
    Z0 = E[features,:].detach().numpy() 
    return Z0

def hardmaxDynamics(alpha, depth, Z0):
    if isinstance(Z0, torch.Tensor):
        Z0 = Z0.detach().numpy()
    if isinstance(alpha, torch.Tensor):
        alpha = alpha.detach().numpy()
    s1 = 1 / (1 + alpha)
    s2 = alpha * s1
    N_rev, n, d = np.shape(Z0)

    Zf = np.zeros((N_rev,n,d))
    for rev in range(N_rev):
        z0 = Z0[rev,:].T
        W = np.zeros((d, n))
        z = np.zeros((d, n, depth))
        f = z0.copy()
        for iter in range(depth):
            # Get dynamics
            for i in range(n):
                IP = np.dot(f[:, i].T, f)
                Pij = np.zeros(n)
                ind = IP == np.max(IP)
                Pij[ind] = 1. / np.sum(ind)
                W[:, i] = s2 * np.sum(Pij * f, axis=1)
            f = s1 * f + W
            z[:, :, iter] = f
        Zf[rev,:] = z[:,:,depth-1].T
    return Zf

def decoder(v, b, Zf):
    v = v.detach().numpy()
    b = b.detach().numpy()
    Z_mean = np.mean(Zf, axis = 1)
    proj = np.dot(Z_mean, v.T) + b #projection to \R via decoder
    pred = 1 / (1 + np.exp(-proj)) # sigmoid
    return pred