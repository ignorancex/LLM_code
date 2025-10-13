import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from torch import nn
from torch.nn.parameter import Parameter
import pandas as pd
import os
import sys
from extensions.common.configs import cmd_args, set_device
from bigg.common.pytorch_util import *
from bigg.model.tree_model import *
from bigg.model.tree_model import *#RecurTreeGen, FenwickTree, batch_tree_lstm3
from datetime import datetime
from bigg.torch_ops.tensor_ops import *
from bigg.common.consts import t_float
import sys




class AdjacencyLSTM(nn.Module):  
  def __init__(self, args):
    '''
    Args Used:
        hidden_dim: An integer indicating the size of hidden dimension of state.
        embed_dim: An integer indicating the embedding dimension of edge existence + weights
        num_layers: Number of layers of neural network
        weighted: Boolean indicating whether graphs are weighted or unweighted
    '''
    super().__init__()
    
    self.hidden_dim = args.hidden_dim
    self.embed_dim = args.embed_dim
    self.num_layers = args.num_layers
    self.weighted = args.weighted
    self.batch_size = None
    self.cross_entropy = torch.nn.functional.binary_cross_entropy
    self.padding = False
    self.max_num_nodes = args.max_num_nodes
    self.constant_nodes = args.constant_nodes
    self.graph_type = args.graph_type
    self.mode = args.wt_mode
    self.wt_range = args.wt_range
    self.wt_scale = args.wt_scale
    self.log_wt = args.log_wt
    self.sm_wt = args.sm_wt
    
    epoch_num = torch.tensor(0, dtype = int)
    self.register_buffer("epoch_num", epoch_num)
    
    self.edge_existence_embedding = Parameter(torch.Tensor(2, 2 * self.embed_dim))
    self.linear_to_probability = nn.Sigmoid()
    
    
    if self.weighted:
        self.hidden_to_mu = MLP(2 * self.hidden_dim, [4 * self.hidden_dim, 1])
        self.hidden_to_var = MLP(2 * self.hidden_dim, [4 * self.hidden_dim, 1])
        self.embed_weight = MLP(1, [2 * self.embed_dim, self.embed_dim])
        
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
        
        self.hidden_to_logit = MLP(2 * self.hidden_dim, [4 * self.hidden_dim, 1])
        self.node_LSTM = nn.LSTM(input_size = 5 * self.embed_dim,
                            hidden_size = 2 * self.hidden_dim,
                            num_layers = self.num_layers,
                            batch_first = True,
                            bias = False)
    
    else:
        self.node_LSTM = nn.LSTM(input_size= 4 * self.embed_dim,
    						hidden_size = 2 * self.hidden_dim,
    						num_layers = self.num_layers,
    						batch_first = True,
    						bias = False)
    	
        self.hidden_to_logit = MLP(2 * self.hidden_dim, [self.hidden_dim, 1])
        
        
    self.softplus = nn.Softplus()
    self.pos_enc = PosEncoding(args.hidden_dim, args.device, 10000)
    self.edge_enc = PosEncoding(args.embed_dim, args.device, 10000)
    
    self.init_node_h = Parameter(torch.Tensor(1, self.hidden_dim))
    self.init_node_c = Parameter(torch.Tensor(1, self.hidden_dim))
    
    self.node_embeddings  = Parameter(torch.Tensor(self.max_num_nodes, self.embed_dim))
    
    glorot_uniform(self)
    
    self.do_mu_0 = args.mu_0
            
    if self.do_mu_0:
        self.mu_0_wt = Parameter(torch.tensor(0, dtype = float))
        self.v_0_wt = Parameter(torch.tensor(1, dtype = float))
    
    
  def forward(self, x_adj):
      # Topology and Padding Indicator Vectors.
      x_adj = x_adj.squeeze(1)
      self.batch_size = x_adj.shape[0]
      ## Forward Pass
      x_adj, num_nodes, max_num_nodes = self.get_num_nodes(x_adj)
      loss_r, loss_w = self.train_model(num_nodes, x_adj, max_num_nodes)
      
      loss_r = loss_r / torch.sum(num_nodes)
      loss_w = loss_w / torch.sum(num_nodes)
      
      return loss_r, loss_w
  
  def get_num_nodes(self, x_adj):
      '''
      Computes number of nodes in batched graphs.
      
      Args Used:
        x_adj: Batched vectorized (un)weighted adjacency vector of lower have adjacency matrix.
      
      Returns:
        x_adj: Batched vectorized (un)weighted adjacency vector with dimension reduced to max_num_nodes
        x_pad: Boolean padding tensor to indicate whether an entry is from the adjacency matrix.
        num_nodes: Tensor of number of nodes in each graph in batch. If no padding is present, num_nodes is 
                   is simply an np array of the number of nodes shared across all graphs.
      '''
      with torch.no_grad():
          pad_id = (x_adj > -1).squeeze(-1)
          adj_len = torch.sum(pad_id, dim = 1)
          num_nodes = 0.5 + torch.sqrt(2 * adj_len + 0.25)
          num_nodes = num_nodes.round().int()
          
          max_num_nodes = num_nodes.max().item()
          max_adj_len = int(0.5 * max_num_nodes * (max_num_nodes - 1))  
          return x_adj, num_nodes, max_num_nodes
  
  
  def standardize_weights(self, x_adj, x_top): 
      if self.log_wt:
          x_adj = torch.log(x_adj + 1 - x_top)
      
      elif self.sm_wt:
          x_adj = torch.log(torch.exp(x_adj) - x_top)
      
      if self.mode == "none" or torch.sum(x_top > 0) == 0:
          return x_adj
      
      if self.epoch_num == 1:
          self.update_weight_stats(x_adj[x_top > 0])
      
      if self.mode == "score":
          x_adj = (x_adj - self.mu_wt) / (self.var_wt**0.5 + 1e-15)
          x_adj = torch.mul(x_adj, x_top)
          
      elif self.mode == "normalize":
          x_adj = -1 + 2 * (x_adj - self.min_wt) / (self.max_wt - self.min_wt + 1e-15)
          x_adj = self.wt_range * torch.mul(x_adj, x_top)
      
      elif self.mode == "scale":
          x_adj = x_adj * self.wt_scale
      
      elif self.mode == "exp":
         x_adj = torch.exp(-self.wt_scale/x_adj)
      
      elif self.mode == "log-exp":
          x_adj = torch.exp(-self.wt_scale / torch.log(1 + x_adj))
      return x_adj
  
  def update_weight_stats(self, edge_feats, initialize=False):
      '''
      Updates necessary global statistics (mean, variance, min, max) of edge_feats per batch
      if standardizing edge_feats prior to MLP embedding. Only performed during the
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
        
        elif self.mode == "normalize":
          batch_max = edge_feats.max()
          batch_min = edge_feats.min()
          self.min_wt = torch.min(batch_min, self.min_wt)
          self.max_wt = torch.max(batch_max, self.max_wt)
        
      if self.do_mu_0 and initialize:
        self.mu_0_wt.data.fill_(self.mu_wt.item())
        self.v_0_wt.data.fill_(self.var_wt.item())
  
  def compute_recon_loss(self, logits, x_top, x_pad):
      '''
      Computes reconstruction loss of graph topology
      
      Args Used:
        probs_out: predicted probabilities for edge existence
        x_top: vectorized adjacency matrix of weighted graph(s)
        x_pad: padding vector, if applicable
      
      Returns:
        recon_loss: reconstruction loss from current forward pass
      '''
      logits = logits.flatten()
      x_top = x_top.flatten().float()
      
      recon_loss = F.binary_cross_entropy_with_logits(logits, x_top, reduction='none')
      
      if x_pad is not None:
          x_pad = x_pad.flatten()
          recon_loss = torch.mul(recon_loss, 1 - x_pad)
      
      recon_loss = torch.sum(recon_loss)
      return recon_loss
  
  def compute_loss_w(self, mus, logvars, x_adj, x_top, threshold = 20):
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
      x_adj = x_adj.flatten()
      x_top = x_top.flatten()
      
      
      mask = (x_top > 0)
      mus = mus[mask]
      logvars = logvars[mask]
      weights = x_adj[mask]
      
      # Compute Loss using a "SoftPlus Normal" Distribution
      loss_w = self.compute_softminus(weights)
      loss_w = torch.square(torch.sub(mus, loss_w))
      loss_w = torch.mul(loss_w, torch.exp(-logvars))
      loss_w = logvars + loss_w
      loss_w = 0.5 * torch.sum(loss_w)
      return loss_w 
  
  def compute_softminus(self, weights, threshold=20):
      '''
      Computes 'softminus' of weights: log(exp(w) - 1). For numerical stability,
      reverts to linear function if w > 20.
      
      Args Used:
        x_adj: adjacency vector at this iteration
        threshold: threshold value to revert to linear function
      
      Returns:
        x_sm: adjacency vector with softminus applied to weight entries
      '''
      x_thresh = (weights <= threshold).float()
      x_sm = torch.mul(x_thresh, weights)
      x_sm = torch.log(torch.exp(x_sm) - x_thresh)
      x_sm = x_sm + torch.mul(weights, 1 - x_thresh)
      return x_sm
  
  def init_state(self, hidden, cell):
      '''
      Initializes state to be fed into edge-level LSTM. For multi-layer LSTMs,
      the initial state consists of zero vectors for all except the final layer.
      
      Args Used:
        hidden: the initial hidden for final layer
        cell: the initial cell for final layer
      
      Returns:
        The initialized state
      '''
      dev = hidden.device
      shape = hidden.shape
      h_last = hidden.unsqueeze(1).repeat(1, self.batch_size, 1)
      c_last = cell.unsqueeze(1).repeat(1, self.batch_size, 1)
      zeros = torch.zeros(self.num_layers - 1, self.batch_size, self.hidden_dim, device=dev)
      hidden = torch.cat([zeros, h_last], dim=0)
      cell = torch.cat([zeros, c_last], dim=0)
      return (hidden, cell)
  
  def set_state(self, row_state, col_state):
      '''
      Sets current state as a concatenation of current ROW and COLUMN state.
      
      Args Used:
        row_state: current state of row i
        col_states: current state of column j
        row_node: row node i
        col_node: column node j
        num_nodes: number of nodes in the graph 
      
      Returns:
        The concatenated row and column state
      '''      
      new_h = torch.cat([row_state[0], col_state[0]], -1)
      new_c = torch.cat([row_state[1], col_state[1]], -1)
      return (new_h, new_c)
    
    
  def split_state(self, state):
      '''
      Splits inputted state into ROW and COLUMN states.
      
      Args Used:
        state: current state to be split
      
      Returns:
        row_state: row_state portion of the inputted state
        col_state: col_state portion of the inputted state
      '''  
      cur_row_h, cur_col_h = torch.split(state[0], self.hidden_dim, -1)
      cur_row_c, cur_col_c = torch.split(state[1], self.hidden_dim, -1)
      
      cur_col_state = (cur_col_h, cur_col_c)
      cur_row_state = (cur_row_h, cur_row_c)
          
      return cur_row_state, cur_col_state
  
  def embed_edges(self, x_cur_it, i_edge, row_node, col_node, num_nodes):
      '''
      Embeds edges for the training stage of the model.
      
      Args Used:
        x_cur_it: vector of current adjacency matrix entry to be embedded
      
      Returns:
         embedded_edges: embedded edges to be fed into edge-level LSTM.
      '''
      embedded_edges = self.edge_existence_embedding[i_edge.squeeze(-1).long()]
      
      ## Embed weights and concatenate (if applicable)
      if self.weighted:            
          x_normalize = self.standardize_weights(x_cur_it, i_edge)
          embedded_weights = self.embed_weight(x_normalize.squeeze(-1)).unsqueeze(1)
          embedded_weights = torch.mul(embedded_weights, i_edge)
          embedded_edges = torch.cat([embedded_edges, embedded_weights], -1)
      
      row_embed = self.node_embeddings[(num_nodes - (row_node + 1)).long()]
      col_embed = self.node_embeddings[(num_nodes - (col_node + 1)).long()]
      embedded_edges = torch.cat([embedded_edges, row_embed.unsqueeze(1), col_embed.unsqueeze(1)], -1)
      return embedded_edges
  
  def get_cur_adj_row(self, x_adj, row_node, col_node):
      idx = int(0.5 * row_node * (row_node - 1) + col_node)
      x_adj_cur = x_adj[:, idx:idx+1, :]
      i_edge = (x_adj_cur > 0).int()
      x_pad_cur = (None if self.constant_nodes else (x_adj_cur == -1).int())
      return x_adj_cur, i_edge, x_pad_cur

  
  def train_model(self, num_nodes, x_adj, max_num_nodes):
    '''
    Forward pass for training.
    
    Args Used:
      num_nodes: number of nodes in each graph
      x_adj: vectorized (un)weighted adjacency matrix of batched graphs
      max_num_nodes: maximum number of nodes possible across all training graphs
    
    Returns:
      loss: computed loss of logits and weight params (if applicable)
    '''
    node_states = {}
    loss_r = 0.0
    loss_w = 0.0
    cur_row_state = self.init_state(self.init_node_h, self.init_node_c)
    
    
    ## Iterate over rows i
    for row_node in range(0, max_num_nodes):
        row_pos = self.pos_enc(num_nodes - row_node)
        cur_row_state = [x + row_pos for x in cur_row_state]
        
        ## Iterate over columns j up to i - 1
        for col_node in range(0, row_node):
            ### Set Current State (i, j) as Concat(row node state i; col node state j)
            cur_col_state = node_states[col_node]
            cur_adj_state = self.set_state(cur_row_state, cur_col_state)
            
            ### Get current entry of adjacency matrix
            x_adj_cur, i_edge, x_pad_cur = self.get_cur_adj_row(x_adj, row_node, col_node)
            
            ### If this row's entire batch is padded, training is complete
            if not self.constant_nodes and torch.sum(x_pad_cur) == self.batch_size:
                return loss_r, loss_w
            
            ### Predict logit and add to loss
            logit = self.hidden_to_logit(cur_adj_state[0][-1])
            loss_r = loss_r + self.compute_recon_loss(logit, i_edge, x_pad_cur)
            
            ### If weighted, predict weight params and add to loss (provided edge exists)
            if self.weighted and i_edge.any() > 0:
                mus = self.hidden_to_mu(cur_adj_state[0][-1])
                logvars = self.hidden_to_var(cur_adj_state[0][-1])
                loss_w = loss_w + self.compute_loss_w(mus, logvars, x_adj_cur, i_edge)
            
            #### Embed edges and update node states via LSTM
            embedded_edges = self.embed_edges(x_adj_cur, i_edge, row_node, col_node, num_nodes)
            _, cur_adj_state = self.node_LSTM(embedded_edges, cur_adj_state)
            cur_row_state, cur_col_state = self.split_state(cur_adj_state)
            
            ### Split next_state into row state i and col state j & update
            node_states[col_node] = cur_col_state
        
        node_states[row_node] = cur_row_state
        
    return loss_r, loss_w
  
  def predict(self, num_nodes, tol = 0.0):
    '''
    Forward pass for sampling. 
    
    Args Used:
      num_nodes: number of nodes in the graph to be generated
      tol: tolerance value to compare generated probability with Uniform(tol, 1-tol) for edge existence sampling
      
    Returns:
      predicted: (un)weighted vectorized adjacency matrix of generated graph
    '''
    predicted = []
    node_states = {}
    
    if self.num_layers > 1:
        init_h = torch.cat([torch.zeros(self.num_layers - 1, self.hidden_dim).to(self.init_node_h.device), self.init_node_h], dim = 0)
        init_c = torch.cat([torch.zeros(self.num_layers - 1, self.hidden_dim).to(self.init_node_c.device), self.init_node_c], dim = 0)
        cur_row_state = (init_h, init_c)
    
    else:
        cur_row_state = (self.init_node_h, self.init_node_c)
    
    cur_row_state = (cur_row_state[0].unsqueeze(1), cur_row_state[1].unsqueeze(1))
    
    for row_node in range(0, num_nodes):
        row_pos = self.pos_enc([num_nodes - row_node])
        cur_row_state = [x + row_pos for x in cur_row_state]
        node_states[row_node] = cur_row_state
        for col_node in range(0, row_node):
            cur_col_state = node_states[col_node]
            cur_adj_state = self.set_state(cur_row_state, cur_col_state)
            
            logit = self.hidden_to_logit(cur_adj_state[0][-1])
            
            prob = self.linear_to_probability(logit)
            p = np.random.uniform(low = 0.0 + tol, high = 1.0 - tol, size = 1)
            i_edge = float(prob.item() > p.item())
            embed_edge = self.edge_existence_embedding[int(i_edge)]
            embed_edge = embed_edge.reshape(1, 1, 2 * self.embed_dim)
            
            if self.weighted and i_edge:
                mu = self.hidden_to_mu(cur_adj_state[0][-1])
                lvar = self.hidden_to_var(cur_adj_state[0][-1])
                sd = torch.exp(0.5 * lvar)
                w = torch.normal(mu, sd)
                w = self.softplus(w)
                
                predicted.append(w.item())
                
                w_embedding = self.standardize_weights(w, torch.tensor([i_edge]).to(w.device))
                w_embedding = self.embed_weight(w_embedding.unsqueeze(0))
                embed_edge = torch.cat([embed_edge, w_embedding], -1)
            
            elif self.weighted:
                embed_edge = torch.cat([embed_edge, torch.zeros(1, 1, self.embed_dim).to(embed_edge.device)], -1)
                predicted.append(i_edge)
            
            else:
                predicted.append(i_edge)
            
            #### Split next_state into row state i and col state j
            row_embed = self.node_embeddings[num_nodes - (row_node + 1)].reshape(1, 1, self.embed_dim)
            col_embed = self.node_embeddings[num_nodes - (col_node + 1)].reshape(1, 1, self.embed_dim)
            embed_edge = torch.cat([embed_edge, row_embed, col_embed], -1)
            
            _, cur_adj_state = self.node_LSTM(embed_edge, cur_adj_state)
            cur_row_state, cur_col_state = self.split_state(cur_adj_state)
            
            ### Column State Updater
            node_states[col_node] = cur_col_state
            node_states[row_node] = cur_row_state   
    return predicted

















































