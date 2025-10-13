import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torch_geometric
from torch_geometric import nn
from torch_geometric.nn import conv
from bigg.common.pytorch_util import glorot_uniform, MLP
from bigg.common.configs import cmd_args, set_device


class GCN(torch.nn.Module):
    def __init__(self, node_embed_dim, embed_dim, out_dim, max_num_nodes):
        super().__init__()
        self.max_num_nodes = max_num_nodes
        self.embed_dim = embed_dim
        self.node_embed_dim = node_embed_dim
        self.out_dim = out_dim
        
        self.conv1 = conv.GCNConv(in_channels = self.node_embed_dim, out_channels = self.embed_dim)
        self.conv2 = conv.GCNConv(in_channels = self.embed_dim, out_channels = self.out_dim)
        self.node_embedding = torch.nn.Embedding(self.max_num_nodes, self.node_embed_dim)
    
    def forward(self, feat_idx, edge_list):
        node_embeddings = self.node_embedding.weight[feat_idx.long()]
        h = self.conv1(node_embeddings, edge_list.long())
        h = F.relu(h)
        h = self.conv2(h, edge_list.long())
        return h


class GCN_Generate(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        ### Hyperparameters
        self.node_embed_dim = args.node_embed_dim
        self.embed_dim = args.embed_dim
        self.embed_dim_wt = args.embed_dim_wt
        self.max_num_nodes = args.max_num_nodes
        self.num_layers = args.rnn_layers
        
        ### GCN Model and GRU 
        self.GCN_mod = GCN(self.node_embed_dim, self.embed_dim, self.embed_dim_wt, self.max_num_nodes)
                
        ### MLPs for mu, logvar, and weight embeddings
        self.hidden_to_mu = MLP(2 * self.embed_dim_wt, [4 * self.embed_dim_wt, 1])
        self.hidden_to_logvar = MLP(2 * self.embed_dim_wt, [4 * self.embed_dim_wt, 1])
        #self.embed_weight = MLP(1, [2 * self.node_embed_dim, self.node_embed_dim])
        
        self.softplus = torch.nn.Softplus()
        
        ### Statistics for weight standardization
        mu_wt = torch.tensor(0, dtype = float)
        var_wt = torch.tensor(1, dtype = float)
        n_obs = torch.tensor(0, dtype = int)
        min_wt = torch.tensor(np.inf, dtype = float)
        max_wt = torch.tensor(-np.inf, dtype = float)
        
        self.register_buffer("mu_wt", mu_wt)
        self.register_buffer("var_wt", var_wt)
        self.register_buffer("n_obs", n_obs)
        self.register_buffer("min_wt", min_wt)
        self.register_buffer("max_wt", max_wt)
        
        self.mode = args.wt_mode
        
        self.log_wt = False
        self.sm_wt = False
        self.wt_range = 1.0
        self.wt_scale = args.wt_scale
        
        self.do_mu_0 = args.mu_0
        
        if self.do_mu_0:
            self.mu_0_wt = Parameter(torch.tensor(0, dtype = float))
            self.s_0_wt = Parameter(torch.tensor(1, dtype = float))
        
        glorot_uniform(self)
        
        
    
    ## Helper functions from LSTM model that are needed (weight loss, standardizing, ...)
    def compute_ll_w(self, mus, logvars, weights):
      '''
      Computes loss of graph weights, if graph is weighted
      
      Args Used:
        mus: model predicted means for each weight
        logvars: model predicted variances for each weight; log-scale
        weights: current weights of graph(s)
      
      Returns:
        loss_w: total loss of graph weights
      '''
      mus = mus.flatten()
      logvars = logvars.flatten()
      weights = weights.flatten()
    
      # Compute Loss using a "SoftPlus Normal" Distribution
      ll = self.compute_softminus(weights)
      ll = torch.square(torch.sub(mus, ll))
      ll = torch.mul(ll, torch.exp(-logvars))
      ll = logvars + ll
      ll = -0.5 * torch.sum(ll)
      return ll
    
    def standardize_edge_feats(self, edge_feats): 
      if self.log_wt:
        edge_feats = torch.log(edge_feats)
      
      elif self.sm_wt:
        edge_feats = torch.log(torch.special.expm1(edge_feats))
      
      if self.mode == "none":
        return edge_feats
      
      if self.mode == "score":
        edge_feats = (edge_feats - self.mu_wt) / (self.var_wt**0.5 + 1e-15)
          
      elif self.mode == "normalize":
        edge_feats = -1 + 2 * (edge_feats - self.min_wt) / (self.max_wt - self.min_wt + 1e-15)
        edge_feats = self.wt_range * edge_feats
      
      elif self.mode == "scale":
        edge_feats = edge_feats * self.wt_scale
      
      elif self.mode == "exp":
        edge_feats = torch.exp(-1/edge_feats)
      
      elif self.mode == "exp-log":
        edge_feats = torch.exp(-1 / torch.log(1 + edge_feats))
        
      return edge_feats
  
    def update_weight_stats(self, edge_feats):
      '''
      Updates global mean and standard deviation of weights per batch if
      standardizing weights prior to MLP embedding. Only performed during the
      first epoch of training.
      
      Args Used:
        weights: weights from current iteration batch
      '''
      
      ## Current training weight statistics
      with torch.no_grad():
          if self.mode == "score":
              ## Current Global Weight Statistics
              mu_n = self.mu_wt
              var_n = self.var_wt
              n = self.n_obs
              
              ## New batch weight statistics
              m = len(edge_feats)
              
              if m > 1:
                  var_m = torch.var(edge_feats)
              
              else:
                  var_m = 0.0
                
              mu_m = torch.mean(edge_feats)
              tot = n + m
              
              ## Update weight statistics
              if tot == 1:
                  self.mu_wt = mu_m
                  self.n_obs = tot
             
              else:
                  new_mu = (n * mu_n + m * mu_m) / tot
                
              new_var_avg = (max(n - 1, 0) * var_n + (m - 1) * var_m)/(tot - 1)
              new_var_resid = n * m * (mu_n - mu_m)**2 / (tot * (tot - 1))
              new_var = new_var_avg + new_var_resid
              
              ## Save
              self.mu_wt = new_mu
              self.var_wt = new_var
              self.n_obs += m
            
              if self.do_mu_0:
                  self.mu_0_wt = Parameter(self.mu_0_wt * 0 + self.mu_wt.item())
                  self.s_0_wt = Parameter(self.s_0_wt * 0 + self.var_wt.item())
          
          elif self.mode == "normalize":
              batch_max = edge_feats.max()
              batch_min = edge_feats.min()
              self.min_wt = torch.min(batch_min, self.min_wt)
              self.max_wt = torch.max(batch_max, self.max_wt)

        
    
    def compute_softminus(self, edge_feats, threshold = 20):
        '''
        Computes 'softminus' of weights: log(exp(w) - 1). For numerical stability,
        reverts to linear function if w > 20.
        
        Args Used:
          x_adj: adjacency vector at this iteration
          threshold: threshold value to revert to linear function
        
        Returns:
          x_sm: adjacency vector with softminus applied to weight entries
        '''
        x_thresh = (edge_feats <= threshold).float()
        x_sm = torch.log(torch.special.expm1(edge_feats))
        x_sm = torch.mul(x_sm, x_thresh)
        x_sm = x_sm + torch.mul(edge_feats, 1 - x_thresh)
        return x_sm
    
    def forward(self, feat_idx, edge_list, batch_weight_idx):
        h = self.GCN_mod.forward(feat_idx, edge_list[0:2, :])
        
        
        edges = batch_weight_idx[:, 0:2].long()
        weights = batch_weight_idx[:, 2:3]
        
        batch_idx = edge_list[2:3, :].flatten()
        
        nodes = h[edges].flatten(1)
        combined = nodes
        
        mu_wt = self.hidden_to_mu(combined)
        logvar_wt = self.hidden_to_logvar(combined)
        
        if self.do_mu_0:
            mu_wt = mu_wt + self.mu_0_wt
            logvar_wt = logvar_wt  + torch.log(self.s_0_wt)
        
        ll_wt = self.compute_ll_w(mu_wt, logvar_wt, weights)
        return ll_wt
    
    def sample(self, num_nodes, edge_list):
        feat_idx = torch.arange(num_nodes, device=edge_list.device)
        h = self.GCN_mod.forward(feat_idx, edge_list.t())  # [num_nodes, embed_dim]
        edges = edge_list.long()  # [num_edges, 2]
        
        # Index both ends of the edge: [num_edges, 2 * embed_dim]
        nodes = h[edges]  # [num_edges, 2, embed_dim]
        combined = nodes.flatten(1)  # [num_edges, 2 * embed_dim]
        
        mu_wt = self.hidden_to_mu(combined)         # [num_edges]
        logvar_wt = self.hidden_to_logvar(combined) # [num_edges]
        
        if self.do_mu_0:
            mu_wt = mu_wt + self.mu_0_wt
            logvar_wt = logvar_wt + torch.log(self.s_0_wt)
        
        std_wt = torch.exp(0.5 * torch.clamp(logvar_wt, max=10))
        weight = torch.normal(mu_wt, std_wt)  # [num_edges]
        weights = self.softplus(weight) / self.wt_scale  # [num_edges]
        
        # Append weights as third column
        weighted_edges = torch.cat([edge_list, weights], dim=-1)  # [num_edges, 3]
        return weighted_edges




        


































