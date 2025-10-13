import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from torch.nn.parameter import Parameter
from torch.nn import Module
from extensions.BiGG_E.model_extensions.tree_functions import *
from extensions.BiGG_E.model_extensions.gcn_build import *
from bigg.common.pytorch_util import glorot_uniform, _glorot_uniform
from bigg.model.tree_model import RecurTreeGen
from extensions.BiGG_E.model_extensions.util_extension.pytorch_util_extension import *


## Adapted From BiGG to support BiGG-E
## SOURCE: https://github.com/google-research/google-research/blob/c097eb6c850370c850eb7a90ab8a22fd2e1c730a/bigg/bigg/extension/customized_models.py#L34

class BiGGExtension(RecurTreeGenExtension):
    def __init__(self, args):
        super().__init__(args)
        if args.has_edge_feats:
            self.method = args.method
            self.device = args.device
            self.sampling_method = args.sampling_method
            self.g_type = args.g_type
            self.embed_dim = args.embed_dim
            self.embed_dim_wt = args.embed_dim_wt
            self.use_mlp = args.use_mlp
            self.debug = args.debug
            
            assert self.sampling_method in ['gamma', 'lognormal', 'softplus']
            assert self.method in ['BiGG-MLP', 'BiGG-E']
            
            if self.method == "BiGG-MLP":
                assert args.rnn_layers == 1
                assert args.bits_compress <= 0
                self.edgelen_encoding_wt = MLP(1, [2 * args.embed_dim, args.embed_dim], dropout = args.wt_drop)
                self.edgelen_mean_wt = MLP(args.embed_dim, [2 * args.embed_dim, 1], dropout = args.wt_drop)
                self.edgelen_lvar_wt = MLP(args.embed_dim, [2 * args.embed_dim, 1], dropout = args.wt_drop)
                self.joint_update_wt = None
                self.joint_merge_has_ch = None
                self.joint_merge_lr = None
                            
            elif self.method == "BiGG-E":
                self.edgelen_mean_wt = MLP(args.embed_dim_wt, [2 * args.embed_dim_wt, 1], dropout = args.wt_drop)
                self.edgelen_lvar_wt = MLP(args.embed_dim_wt, [2 * args.embed_dim_wt, 1], dropout = args.wt_drop)
                self.weight_tree = FenwickTreeExtension(args, weights=True)
                self.leaf_LSTM_wt = nn.LSTMCell(1, args.embed_dim_wt)
                self.joint_update_wt = BinaryTreeLSTMCellWt(args.embed_dim_wt, args.embed_dim, use_mlp = self.use_mlp)
                self.joint_merge_has_ch = BinaryTreeLSTMCellWt(args.embed_dim, args.embed_dim_wt, use_mlp = self.use_mlp)
                self.joint_merge_lr = BinaryTreeLSTMCellWt(args.embed_dim, args.embed_dim_wt, use_mlp = self.use_mlp)
                
                if args.embed_dim_wt != args.embed_dim:
                    self.proj_wt_c = nn.Linear(args.embed_dim_wt, args.embed_dim)
                    self.proj_top_c = nn.Linear(args.embed_dim, args.embed_dim_wt)
            
            glorot_uniform(self)
        
            if args.has_edge_feats and self.method == "BiGG-E" and args.embed_dim_wt != args.embed_dim:
                self.proj_wt_c.bias.requires_grad = False
                self.proj_top_c.bias.requires_grad = False
            
            mu_wt = torch.tensor(0, dtype = float)
            var_wt = torch.tensor(1, dtype = float)
            n_obs = torch.tensor(0, dtype = int)
            min_wt = torch.tensor(torch.inf, dtype = float)
            max_wt = torch.tensor(-torch.inf, dtype = float)
            self.do_mu_0 = args.mu_0
            
            if self.do_mu_0:
                self.mu_0_wt = Parameter(torch.tensor(0, dtype = float))
                self.v_0_wt = Parameter(torch.tensor(1, dtype = float))
            
            self.register_buffer("mu_wt", mu_wt)
            self.register_buffer("var_wt", var_wt)
            self.register_buffer("n_obs", n_obs)
            self.register_buffer("min_wt", min_wt)
            self.register_buffer("max_wt", max_wt)
            self.mode = args.wt_mode
            self.dynam_score = args.dynam_score
            self.wt_scale = args.wt_scale
        
        else:
            self.debug = args.debug
            self.method = 'None'
            self.sampling_method = 'None'
            self.g_type = args.g_type
            self.embed_dim = args.embed_dim
            self.joint_update_wt = None
            self.joint_merge_has_ch = None
            self.joint_merge_lr = None

 
    def standardize_edge_feats(self, edge_feats): 
      if self.mode == "none":
        return edge_feats
      
      if self.mode == "score":
        if self.dynam_score and self.do_mu_0:
            edge_feats = (edge_feats - self.mu_0_wt) / (self.v_0_wt**0.5 + 1e-15)
        else:
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
  
    def update_weight_stats(self, edge_feats, initialize=False):
      '''
      Updates necessary global statistics (mean, variance, min, max) of edge_feats per batch
      if standardizing edge_feats prior to MLP embedding. Only performed during the
      first epoch of training.
      
      Args Used:
        edge_feats: edge_feats from current iteration batch
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

    def embed_edge_feats(self, edge_feats, list_num_edges=None, lv_lists=None):
        B = edge_feats.shape[0]
        edge_feats_normalized = self.standardize_edge_feats(edge_feats)
        
        if self.method == "BiGG-MLP":
            edge_embed = self.edgelen_encoding_wt(edge_feats_normalized)
            edge_embed = edge_embed
        
        elif self.method == "BiGG-E":
            edge_embed = self.leaf_LSTM_wt(edge_feats_normalized)
            
            if list_num_edges is None:
                edge_embed = self.weight_tree(edge_embed)
            
            else:
                edge_embed = self.weight_tree.forward_train(edge_embed, list_num_edges, lv_lists=lv_lists)
        
        return edge_embed
    
    def predict_edge_feats(self, state, edge_feats=None, cur_batch_idx=None):
        """
        Args:
            state: tuple of (h=N x embed_dim, c=N x embed_dim), the current state
            edge_feats: N x feat_dim or None
        Returns:
            likelihood of edge_feats under current state,
            and, if edge_feats is None, then return the prediction of edge_feats
            else return the edge_feats as it is
        """
        h, _ = state
        
        if edge_feats is None:
            ll = 0
            mus, lvars = self.edgelen_mean_wt(h), self.edgelen_lvar_wt(h)
            
            if self.sampling_method == "softplus": 
                if self.do_mu_0:
                    pred_mean = mus + self.mu_0_wt
                    pred_lvar = lvars  + torch.log(self.v_0_wt)
                
                else:
                    pred_mean = mus
                    pred_lvar = lvars
                
                pred_sd = torch.exp(0.5 * pred_lvar)
                edge_feats = torch.normal(pred_mean, pred_sd)
            
            elif self.sampling_method  == "lognormal":
                if self.do_mu_0:
                    pred_mean = mus + self.mu_0_wt
                    pred_lvar = lvars + torch.log(self.v_0_wt)
                
                else:
                    pred_mean = mus
                    pred_lvar = lvars
                
                pred_sd = torch.exp(0.5 * pred_lvar)
                edge_feats = torch.normal(pred_mean, pred_sd)
            
            elif self.sampling_method  == "gamma": 
                if self.do_mu_0:
                    loga = 2 * torch.log(self.mu_0_wt) - torch.log(self.v_0_wt) + mus
                    logb = torch.log(self.mu_0_wt) - torch.log(self.v_0_wt) + lvars
                
                else:
                    loga = mus
                    logb = lvars
                
                a = torch.exp(loga)
                b = torch.exp(logb)
                edge_feats = torch.distributions.gamma.Gamma(a, b).sample()
            
            return ll, edge_feats
                
        else:
            mus, lvars = self.edgelen_mean_wt(h), self.edgelen_lvar_wt(h)    
            
            if self.sampling_method  == "softplus":
                if self.do_mu_0:
                    mus = mus + self.mu_0_wt
                    lvars = lvars + torch.log(self.v_0_wt)
                
                var = torch.exp(lvars)
                diff_sq = torch.square(torch.sub(mus, edge_feats))
                diff_sq = torch.div(diff_sq, var + 1e-15)
                ll = lvars + diff_sq
                ll = -0.5 * ll
            
            elif self.sampling_method  == "lognormal":
                if self.do_mu_0:
                    mus = mus + self.mu_0_wt
                    lvars = lvars + torch.log(self.v_0_wt)
                
                var = torch.exp(lvars)
                
                ll = edge_feats - mus
                ll = torch.square(ll)
                ll = torch.div(ll, var)
                ll = ll + lvars
                ll = -0.5 * ll
            
            elif self.sampling_method  == "gamma":
                if self.do_mu_0:
                    loga = 2 * torch.log(self.mu_0_wt) - torch.log(self.v_0_wt) + mus
                    logb = torch.log(self.mu_0_wt) - torch.log(self.v_0_wt) + lvars
                
                else:
                    loga = mus
                    logb = lvars
                
                a = torch.exp(loga)
                b = torch.exp(logb)
                log_edge_feats = torch.log(edge_feats)
                ll = torch.mul(a, logb)
                ll = ll - torch.lgamma(a)
                ll = ll + torch.mul(a - 1, log_edge_feats)
                ll = ll - torch.mul(b, edge_feats)
            
            ll = torch.sum(ll)
        return ll, edge_feats

 
class BiGGWithGCN(RecurTreeGen):
    def __init__(self, args):
        super().__init__(args)
        self.gcn_mod = GCN_Generate(args)
        self.method = "None"
        self.joint_update_wt = None
        self.joint_merge_has_ch = None
        self.joint_merge_lr = None
        self.device = args.device
        self.debug = args.debug
        
    def forward_train2(self, batch_indices, feat_idx, edge_list, batch_weight_idx, edge_feat_info):
        ll_top, _ = self.forward_train(batch_indices) #, edge_feat_info=edge_feat_info)
        ll_wt = self.gcn_mod.forward(feat_idx, edge_list, batch_weight_idx)
        return ll_top, ll_wt
    
    def sample2(self, num_nodes, display=None):
        # --- Topology generation (BiGG stage) ---
        _, pred_edges, _, _, _ = self.forward(node_end=num_nodes, display=display)
        
        # --- Weight sampling (GCN stage) ---
        fix_edges = [(min(e1, e2), max(e1, e2)) for e1, e2 in pred_edges]
        pred_edge_tensor = torch.tensor(fix_edges, device=self.device)
        pred_weighted_tensor = self.gcn_mod.sample(num_nodes, pred_edge_tensor)
        
        return pred_edges, pred_weighted_tensor
    
    















