import math
import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np

from typing import Optional
from omegaconf import DictConfig

def masked_softmax(tensor, mask):
    mask = mask.bool()  # Convert the mask to boolean data type if it's not already
    tensor_masked = tensor.masked_fill(mask, float('-inf'))  # Apply mask before exponentiation
    softmax = torch.nn.functional.softmax(tensor_masked, dim=-1)
    return softmax

def get_lambda(alpha=1.0):
    '''Return lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.
    return lam

def get_lambda_per_sample(batch_size, alpha=1.0):
    '''Return lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha, size=batch_size)
    else:
        lam = np.ones(batch_size)
    return lam

def to_one_hot(inp,num_classes):
    y_onehot = torch.FloatTensor(inp.size(0), num_classes)
    y_onehot.zero_()
    y_onehot.scatter_(1, inp.unsqueeze(1).data.cpu(), 1)
    return y_onehot

def mixup_process(out, target_reweighted, lam, target=None, mixup_type="random"):
    batch_size, num_features, feature_dim = out.shape
    indices = np.arange(batch_size)
    
    if mixup_type == "cosine_sim":
        # Calculate cosine similarity between samples in the batch
        out_normalized = out.cpu().detach().numpy().reshape(batch_size, -1) / np.linalg.norm(out.cpu().detach().numpy().reshape(batch_size, -1), axis=-1, keepdims=True)
        similarity_matrix = np.dot(out_normalized, out_normalized.T)
        np.fill_diagonal(similarity_matrix, -1)
        most_similar_indices = np.argmax(similarity_matrix, axis=1)

        # Update indices for mixup
        indices = most_similar_indices
        
    elif mixup_type == "class_aware":
        if target is None:
            raise ValueError("Labels must be provided for class-aware mixup.")
       
        for i in range(batch_size):
            same_class_indices = np.where(target.cpu() == target.cpu()[i])[0]
            indices[i] = np.random.choice(same_class_indices)
                
    elif mixup_type == "random":
        indices = np.random.permutation(batch_size)
    
    else:
        raise ValueError("Invalid mixup_type. Choose from ['random', 'cosine_similarity', 'class_aware'].")

    out = out * lam + out[indices] * (1 - lam)
    target_shuffled_onehot = target_reweighted[indices]
    target_reweighted = target_reweighted * lam + target_shuffled_onehot * (1 - lam)
    
    return out, target_reweighted

def mixup_process_per_sample(out, target_reweighted, lam, target=None, mixup_type="random"):
    batch_size, num_features, feature_dim = out.shape
    indices = np.arange(batch_size)
    
    if mixup_type == "cosine_sim":
        out_normalized = out.cpu().detach().numpy().reshape(batch_size, -1) / np.linalg.norm(out.cpu().detach().numpy().reshape(batch_size, -1), axis=-1, keepdims=True)
        similarity_matrix = np.dot(out_normalized, out_normalized.T)
        np.fill_diagonal(similarity_matrix, -1)
        most_similar_indices = np.argmax(similarity_matrix, axis=1)

        indices = most_similar_indices
        
    elif mixup_type == "class_aware":
        if target is None:
            raise ValueError("Labels must be provided for class-aware mixup.")
       
        for i in range(batch_size):
            same_class_indices = np.where(target.cpu() == target.cpu()[i])[0]
            indices[i] = np.random.choice(same_class_indices)
                
    elif mixup_type == "random":
        indices = np.random.permutation(batch_size)
    
    else:
        raise ValueError("Invalid mixup_type. Choose from ['random', 'cosine_similarity', 'class_aware'].")

    lam = lam.view(batch_size, 1, 1)
    out = out * lam + out[indices] * (1 - lam)
    target_shuffled_onehot = target_reweighted[indices]
    lam = lam.view(batch_size, 1)
    target_reweighted = target_reweighted * lam + target_shuffled_onehot * (1 - lam)
   
    return out, target_reweighted

def cantor_diagonal(p: int, q: int):
    return (p+q)*(1+p+q)/2+q

def xavier_normal_embedding_init(m):
    if isinstance(m, nn.Embedding):
        embedding_dim = m.embedding_dim
        init.xavier_normal_(m.weight.data)
        # Divide the weights by the sqrt of the embedding dimension to match Xavier initialization
        m.weight.data /= math.sqrt(embedding_dim)

class PositionalEncoderFactory:
    def __init__(
        self,
        type: str,
        learned: bool, 
        options: Optional[DictConfig] = None,
    ):

        if type == "1d":
            if learned:
                self.pos_encoder = LearnedPositionalEncoding(options.dim, options.dropout, options.max_seq_len)
            else:
                self.pos_encoder = PositionalEncoding(options.dim, options.dropout, options.max_seq_len)
        elif type == "2d":
            if learned:
                if options.agg_method == "concat":
                    self.pos_encoder = ConcatPositionalEmbedding2d(options.tile_size, options.dim, options.max_seq_len, options.max_nslide)
                elif options.agg_method == "self_att":
                    self.pos_encoder = PositionalEmbedding2d(options.tile_size, options.dim, options.max_seq_len)
            else:
                if options.agg_method == "concat":
                    self.pos_encoder = ConcatPositionalEncoding2d(options.dim,options.dropout, options.max_seq_len, options.max_nslide)
                elif options.agg_method == "self_att":
                    self.pos_encoder = PositionalEncoding2d(options.dim, options.dropout, options.max_seq_len)
        else:
           raise ValueError(f"cfg.model.slide_pos_embed.type ({type}) not supported")

    def get_pos_encoder(self):
        return self.pos_encoder


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class LearnedPositionalEncoding(nn.Module): 
    """ Positional encoding layer
    Parameters
    ----------
    dropout : float
    Dropout value.
    num_embeddings : int
    Number of embeddings to train.
    hidden_dim : int
    Embedding dimensionality
    """
    
    def __init__(self, hidden_dim: int, dropout: int = 0.1, max_len: int = 5000):
        super(LearnedPositionalEncoding, self).__init__()
        
        self.weight = nn.Parameter(torch.Tensor(max_len, hidden_dim))
        self.dropout = nn.Dropout(p=dropout)
        self.hidden_dim = hidden_dim
        
        self.reset_parameters()
    
    def reset_parameters(self):
        torch.nn.init.xavier_normal_(self.weight)
    
    def forward(self, x):
        batch_size, seq_len = x.size()[:2]
        embeddings = self.weight[:seq_len, :].view(1, seq_len, self.hidden_dim)
        x = x + embeddings
        return self.dropout(x)
    
class PositionalEncoding2d(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 1000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        X = torch.arange(max_len)
        Y = torch.arange(max_len)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, max_len, d_model)
        for x in X:
            for y in Y:
                position = cantor_diagonal(x,y)
                pe[x, y, 0::2] = torch.sin(position * div_term)
                pe[x, y, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x, coords):
        """
        Args:
            x: Tensor, shape [seq_len, embedding_dim]
            coords: Tensor, shape [seq_len, 2]
        """
        x = x + self.pe[coords[:,0], coords[:,1]]
        return self.dropout(x)


class ConcatPositionalEncoding2d(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 1000, max_nslide: int = 10):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        X = torch.arange(max_len)
        Y = torch.arange(max_len)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, max_len, d_model)
        for x in X:
            for y in Y:
                position = cantor_diagonal(x,y)
                pe[x, y, 0::2] = torch.sin(position * div_term)
                pe[x, y, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

        slide_pos = torch.arange(max_nslide).unsqueeze(1)
        slide_div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        slide_pe = torch.zeros(max_nslide, 1, d_model)
        slide_pe[:, 0, 0::2] = torch.sin(slide_pos * slide_div_term)
        slide_pe[:, 0, 1::2] = torch.cos(slide_pos * slide_div_term)
        self.register_buffer('slide_pe', slide_pe.squeeze(1))

    def forward(self, x, coords):
        """
        Args:
            x: Tensor, shape [seq_len, embedding_dim]
            coords: Tensor, shape [seq_len, 3]
        """
        slide_idx, coord = coords[:, 0], coords[:, 1:]
        x = x + self.pe[coord[:,0], coord[:,1]] + self.slide_pe[slide_idx]
        return self.dropout(x)


class PositionalEmbedding(nn.Module):

    def __init__(self, dim: int, max_len: int = 3000):
        super().__init__()
        self.pos_ids = torch.arange(max_len)
        self.embedder = nn.Embedding(max_len, dim)
        nn.init.normal_(self.embedder.weight, std=0.02)
        # xavier_normal_embedding_init(self.embedder)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        batch_size, seq_length, embedding_dim = x.shape
    
        # create positional ids and embeddings for each sequence in the batch
        position_ids = self.pos_ids[:seq_length].unsqueeze(0).repeat(batch_size, 1).to(x.device)
        position_embeddings = self.embedder(position_ids)
        
        # add positional embeddings to input tensor
        x = x + position_embeddings

        return x

class PositionalEmbedding2d(nn.Module):

    def __init__(self, tile_size: int, dim: int, max_len: int = 512):
        super().__init__()
        self.tile_size = tile_size
        self.embedder_x = nn.Embedding(max_len, dim//2)
        self.embedder_y = nn.Embedding(max_len, dim//2)

    def get_grid_values(self, coords: np.ndarray):
        m = coords.min()
        grid_coords = torch.div(coords-m, self.tile_size, rounding_mode='floor')
        return grid_coords

    def forward(self, x, coords):
        """
        Args:
            x: Tensor, shape [seq_len, embedding_dim]
            coords: Tensor, shape [seq_len, 3]
        """
        _, coord = coords[:, 0], coords[:, 1:]
        coord_x = self.get_grid_values(coord[:,0])
        coord_y = self.get_grid_values(coord[:,1])
        embedding_x = self.embedder_x(coord_x)
        embedding_y = self.embedder_y(coord_y)
        position_embedding = torch.cat([embedding_x, embedding_y], dim=1)
        x += position_embedding
        return x


class ConcatPositionalEmbedding2d(nn.Module):

    def __init__(self, tile_size: int, dim: int, max_len: int = 512, max_nslide: int = 10):
        super().__init__()
        self.tile_size = tile_size
        self.embedder_x = nn.Embedding(max_len, dim//2)
        self.embedder_y = nn.Embedding(max_len, dim//2)
        self.embedder_slide_pos = nn.Embedding(max_nslide, dim)

    def get_grid_values(self, coords: np.ndarray):
        m = coords.min()
        grid_coords = torch.div(coords-m, self.tile_size, rounding_mode='floor')
        return grid_coords

    def forward(self, x, coords):
        """
        Args:
            x: Tensor, shape [seq_len, embedding_dim]
            coords: Tensor, shape [seq_len, 3]
        """
        slide_idx, coord = coords[:, 0], coords[:, 1:]
        coord_x = self.get_grid_values(coord[:,0])
        coord_y = self.get_grid_values(coord[:,1])
        embedding_x = self.embedder_x(coord_x)
        embedding_y = self.embedder_y(coord_y)
        position_embedding = torch.cat([embedding_x, embedding_y], dim=1)
        slide_pos_embedding = self.embedder_slide_pos(slide_idx)
        x = x + position_embedding + slide_pos_embedding
        return x
    
class Attn_Net_Gated(nn.Module):
    def __init__(self, L=1024, D=256, dropout=False, num_classes=1):
        r"""
        Attention Network with Sigmoid Gating (3 fc layers)
        args:
            L (int): input feature dimension
            D (int): hidden layer dimension
            dropout (bool): whether to apply dropout (p = 0.25)
            num_classes (int): number of classes
        """
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [nn.Linear(L, D), nn.Tanh()]

        self.attention_b = [nn.Linear(L, D), nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        self.attention_c = nn.Linear(D, num_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x num_classes
        return A, x
