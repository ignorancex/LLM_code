import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from torch.nn.parameter import Parameter
from bigg.common.pytorch_util import glorot_uniform
from bigg.model.tree_model import *
from extensions.BiGG_E.model_extensions.util_extension.pytorch_util_extension import *
from extensions.BiGG_E.model_extensions.util_extension.train_util import *


# --------------------------------------------------------------
# Adapted functions for BiGG-E originally from BiGG (Google Research):
# https://github.com/google-research/google-research/blob/c097eb6c850370c850eb7a90ab8a22fd2e1c730a/ugsl/input_layer.py#L103
# Functions adapted: FenwickTree, RecurTreeGen
# Copyright (c) Google LLC
# Licensed under the Apache License 2.0
# --------------------------------------------------------------

class FenwickTreeExtension(FenwickTree):
    def __init__(self, args, weights=False):
        super(FenwickTreeExtension, self).__init__(args)
        self.method = args.method
        self.embed_dim = args.embed_dim
        self.use_mlp = args.use_mlp
        self.weights = weights
        self.embed_dim = args.embed_dim_wt
        self.init_h0_wt = Parameter(torch.Tensor(1, self.embed_dim))
        self.init_c0_wt = Parameter(torch.Tensor(1, self.embed_dim))
        self.merge_cell_wt = BinaryTreeLSTMCellWt(self.embed_dim, self.embed_dim, use_mlp = self.use_mlp)
        self.summary_cell_wt = BinaryTreeLSTMCellWt(self.embed_dim, self.embed_dim, use_mlp = self.use_mlp)
        del self.init_h0
        del self.init_c0
        del self.merge_cell
        del self.summary_cell
        
        glorot_uniform(self)
    
    def forward(self, new_state=None, print_it=False):
        if new_state is None:
            if len(self.list_states) == 0:
                return (self.init_h0_wt, self.init_c0_wt)
        else:
            self.append_state(new_state, 0)
        pos = 0
        while pos < len(self.list_states):
            if len(self.list_states[pos]) >= 2:
                lch_state, rch_state = self.list_states[pos]  # assert the length is 2
                new_state = self.merge_cell_wt(lch_state, rch_state)
                self.list_states[pos] = []
                self.append_state(new_state, pos + 1)
            pos += 1
        state = None
        for pos in range(len(self.list_states)):
            if len(self.list_states[pos]) == 0:
                continue
            cur_state = self.list_states[pos][0]
            if state is None:
                state = cur_state
            else:
                state = self.summary_cell_wt(state, cur_state)
        return state

    def forward_train(self, edge_feats_init_embed, list_num_edges, lv_lists=None):
        # embed row tree
        list_indices = get_list_indices(list_num_edges)
        edge_embeds = [edge_feats_init_embed]
        
        for i, all_ids in enumerate(list_indices):
            fn_ids = lambda x: all_ids[x]
            new_states = batch_tree_lstm3(None, None, h_buf=edge_embeds[-1][0], c_buf=edge_embeds[-1][1], h_past=None, c_past=None, fn_all_ids=fn_ids, cell=self.merge_cell_wt)
            edge_embeds.append(new_states)
        h_list, c_list = zip(*edge_embeds)
        joint_h = torch.cat(h_list, dim=0)
        joint_c = torch.cat(c_list, dim=0)

        # get history representation
        batch_lv_list = get_batch_lv_lists(list_num_edges, lv_lists)
        init_select, all_ids, last_tos = prepare_batch(batch_lv_list)        
        cur_state = (joint_h[init_select], joint_c[init_select])
        
        hist_rnn_states = []
        hist_froms = []
        hist_tos = []
        for i, (done_from, done_to, proceed_from, proceed_input) in enumerate(all_ids):
            hist_froms.append(done_from)
            hist_tos.append(done_to)
            hist_rnn_states.append(cur_state)

            next_input = joint_h[proceed_input], joint_c[proceed_input]
            sub_state = cur_state[0][proceed_from], cur_state[1][proceed_from]
            cur_state = self.summary_cell_wt(sub_state, next_input)
        hist_rnn_states.append(cur_state)
        hist_froms.append(None)
        hist_tos.append(last_tos)
        hist_h_list, hist_c_list = zip(*hist_rnn_states)
        edge_h = multi_index_select(hist_froms, hist_tos, *hist_h_list)
        edge_c = multi_index_select(hist_froms, hist_tos, *hist_c_list)
        edge_embeddings = (edge_h, edge_c)
        return edge_embeddings



class RecurTreeGenExtension(RecurTreeGen):
    def __init__(self, args):
        super(RecurTreeGenExtension, self).__init__(args)
        self.method = args.method
        self.leaf_h0 = Parameter(torch.Tensor(1, args.embed_dim))
        self.leaf_c0 = Parameter(torch.Tensor(1, args.embed_dim))
        self.empty_h0 = Parameter(torch.Tensor(1, args.embed_dim))
        self.empty_c0 = Parameter(torch.Tensor(1, args.embed_dim))
        self.method = args.method
        self.use_mlp = args.use_mlp
        self.row_tree = FenwickTree(args)
        if self.method == "BiGG-E":
            self.row_tree.has_edge_feats = False
        if self.method == "BiGG-MLP":
            self.bits_compress = 0
        else:
            self.bits_compress = args.bits_compress
        
        if self.bits_compress > 0:
            self.bit_rep_net = BitsRepNet(args)
            del self.leaf_h0
            del self.leaf_c0
            del self.empty_h0
            del self.empty_c0
        self.m_l2r_cell = BinaryTreeLSTMCell(args.embed_dim)
        self.lr2p_cell = BinaryTreeLSTMCell(args.embed_dim)
        self.m_cell_topdown = nn.LSTMCell(args.embed_dim, args.embed_dim)
        self.m_cell_topright = nn.LSTMCell(args.embed_dim, args.embed_dim)
        self.cached_indices = {}
        glorot_uniform(self)
        
    def get_empty_state(self):
        if self.bits_compress > 0:
            return self.bit_rep_net([], 1)
        else:
            return (self.empty_h0, self.empty_c0)
    
    def get_merged_prob(self, top_state, wt_state, prob_func=None, merge_func=None):
        top_state_last = (top_state[0], top_state[1])    
        wt_state_last = (wt_state[0], wt_state[1])
        
        if prob_func is None:
            if self.embed_dim != self.embed_dim_wt:
                top_state_last = (top_state_last[0], self.proj_top_c(top_state_last[1]))
                        
            state_update = merge_func(top_state_last, wt_state_last)
            return state_update
        
        else:
            if self.embed_dim != self.embed_dim_wt:
                wt_state_last = (wt_state_last[0], self.proj_wt_c(wt_state_last[1]))
            state_update = merge_func(top_state_last, wt_state_last)
            logit = prob_func(state_update[0])
            return logit
    
    def CacheTopdownIdx(self, train_graphs, list_last_edges):
        graph_ids = list(range(len(train_graphs)))
        for g_id in graph_ids:
            TreeLib.PrepareMiniBatch([g_id], None, -1, None)
            all_ids = TreeLib.PrepareTreeEmbed()
            left_idx, right_idx = self.GetMostRecentWeight(max_depth = len(all_ids) + 1, batch_last_edges=list_last_edges[g_id])
            self.cached_indices[g_id] = (left_idx, right_idx)
    
    def GetMostRecentWeight(self, max_depth, batch_last_edges=None):
        if self.bits_compress > 0:
            max_d_bin = TreeLib.lib.MaxBinFeatDepth()
            max_d_tree = TreeLib.lib.MaxTreeDepth()
            max_depth = max_d_bin + max_depth - (max_d_tree + 1)
        
        most_recent_edge_list = [None] * max_depth
        parent_indices = [None] * max_depth
        is_lch_list = [None] * max_depth
        
        for d in range(max_depth - 1, -1, -1):
            cur_lv_nonleaf = TreeLib.QueryNonLeaf(d)
            cur_lv_edge, _ = TreeLib.GetEdgeAndLR(d)
            
            if d == max_depth - 1:
                cur_weights = cur_lv_edge
            
            else:
                cur_weights = np.zeros(len(cur_lv_nonleaf))
                cur_weights[~cur_lv_nonleaf] = cur_lv_edge
                cur_weights[cur_lv_nonleaf] = mre
            
            if d != max_depth - 1:
                cur_is_left, _ =  TreeLib.GetChLabel(-1, d)
                cur_is_right, _ =  TreeLib.GetChLabel(1, d)
            
            else:
                cur_is_left = None
                cur_is_right = None
            
            if d == 0:
                left_idx = [None] * max_depth
                right_idx = [None] * max_depth
                
                for lv in range(0, max_depth):
                    cur_left_states = None
                    cur_right_states = None
                    cur_par_idx = parent_indices[lv]
                    cur_edge = most_recent_edge_list[lv]
                    cur_is_lch = is_lch_list[lv]
                    
                    if lv == 0:
                        if batch_last_edges is None:
                            cur_left_states = np.array([-1] * len(cur_edge))
                        
                        else:
                            has_ch, _ = TreeLib.GetChLabel(0, dtype=bool)
                            cur_lv_nonleaf = TreeLib.QueryNonLeaf(lv)
                            cur_left_states = batch_last_edges[has_ch][cur_lv_nonleaf]
                            assert len(cur_left_states) == len(cur_edge)
                            
                        cur_right_states = np.array([x[0] for x in cur_edge]) 
                        left_idx[d] = cur_left_states
                        right_idx_1 = np.array([x[0] if x[0] != -1 else -1 for x in cur_edge])
                        right_idx[d] = np.array([x if x != -1 else y for x,y in zip(right_idx_1, cur_left_states)])
                        par_left_edge = np.array([x[0] for x in cur_edge])
                        par_left_states = cur_left_states
                        par_right_states = cur_right_states
                    
                    elif cur_edge is not None:
                        cur_left_states = np.array([-1] * len(cur_edge))
                        idx = np.array([~x and (y != -1) for x, y in zip(cur_is_lch, par_left_edge[cur_par_idx])])
                        cur_left_states[~idx] = par_left_states[cur_par_idx[~idx]]
                        cur_left_states[idx] = par_left_edge[cur_par_idx[idx]]
                        cur_right_states = np.array([x[0] for x in cur_edge])
                        par_left_edge = np.array([x[0] for x in cur_edge])
                        par_left_states = cur_left_states
                        par_right_states = cur_right_states
                        left_idx[lv] = cur_left_states
                        right_idx_1 = np.array([x[0] if x[0] != -1 else -1 for x in cur_edge])
                        right_idx[lv] = np.array([x if x != -1 else y for x,y in zip(right_idx_1,cur_left_states)])
                    
                    else:
                        left_idx[lv] = cur_left_states
                        right_idx[lv] = cur_right_states
                return left_idx, right_idx
            
            up_lv_nonleaf = TreeLib.QueryNonLeaf(d - 1)
            up_is_left, _ = TreeLib.GetChLabel(-1, d - 1)
            up_is_right, _ = TreeLib.GetChLabel(1, d - 1)
            
            num_internal_parents = np.sum(up_lv_nonleaf)
            lch = np.array([-1] * num_internal_parents)
            rch = np.array([-1] * num_internal_parents)
            
            up_is_left = lch * (1 - up_is_left) + up_is_left
            up_is_right = rch * (1 - up_is_right) + up_is_right

            lr = np.concatenate([np.array([x, y]) for x,y in zip(up_is_left, up_is_right)])

            is_lch = np.array([True, False]*len(up_is_left))
            is_lch = is_lch[lr != -1]
            is_lch = is_lch[cur_lv_nonleaf]
            is_lch_list[d] = is_lch
            
            lr = lr.astype(np.int32)
            lr[lr == 1] = cur_weights
            lr = lr.reshape(len(up_is_left), 2)
            lch, rch = lr[:, 0], lr[:, 1]
            
            up_level_lr = np.array([[l, r] for l, r in zip(lch, rch)])
            mre = np.array([x[1] if x[1] != -1 else x[0] for x in up_level_lr])
            most_recent_edge_list[d - 1] = up_level_lr
            
            lch_b = (lch > -1)
            rch_b = (rch > -1)
            num_chil = lch_b.astype(int) + rch_b.astype(int)
            idx_list = list(range(len(num_chil)))
            par_idx = np.array([x for i, x in zip(num_chil, idx_list) for _ in range(i)])
            par_idx = par_idx[cur_lv_nonleaf]
            parent_indices[d] = par_idx
            
            up_lv_nonleaf = TreeLib.QueryNonLeaf(d - 1)
            up_is_left, _ = TreeLib.GetChLabel(-1, d - 1)
            up_is_right, _ = TreeLib.GetChLabel(1, d - 1)
            
            num_internal_parents = np.sum(up_lv_nonleaf)
            lch = np.array([-1] * num_internal_parents)
            rch = np.array([-1] * num_internal_parents)
            
            up_is_left = lch * (1 - up_is_left) + up_is_left
            up_is_right = rch * (1 - up_is_right) + up_is_right

            lr = np.concatenate([np.array([x, y]) for x,y in zip(up_is_left, up_is_right)])

            is_lch = np.array([True, False]*len(up_is_left))
            is_lch = is_lch[lr != -1]
            is_lch = is_lch[cur_lv_nonleaf]
            is_lch_list[d] = is_lch
            
            lr = lr.astype(np.int32)
            lr[lr == 1] = cur_weights
            lr = lr.reshape(len(up_is_left), 2)
            lch, rch = lr[:, 0], lr[:, 1]
            
            up_level_lr = np.array([[l, r] for l, r in zip(lch, rch)])
            mre = np.array([x[1] if x[1] != -1 else x[0] for x in up_level_lr])
            most_recent_edge_list[d - 1] = up_level_lr
            
            lch_b = (lch > -1)
            rch_b = (rch > -1)
            num_chil = lch_b.astype(int) + rch_b.astype(int)
            idx_list = list(range(len(num_chil)))
            par_idx = np.array([x for i, x in zip(num_chil, idx_list) for _ in range(i)])
            par_idx = par_idx[cur_lv_nonleaf]
            parent_indices[d] = par_idx
    
    def get_topdown_idx(self, list_num_edges, batch_last_edges, graph_ids, all_ids_len):
        if self.has_edge_feats and self.method == "BiGG-E":
            if len(self.cached_indices) > 0 and len(graph_ids) > 1:
                batch_offset = np.zeros(len(graph_ids), dtype=np.int64)
                batch_offset[1:] = np.cumsum(list_num_edges[:-1])
                left_raw = [self.cached_indices[g_id][0][:-1] for g_id in graph_ids]
                right_raw = [self.cached_indices[g_id][1][:-1] for g_id in graph_ids]
                max_depth = max(len(l) for l in left_raw)
                
                # Preallocate padded and offset lists
                padded_left = [[] for _ in range(max_depth)]
                padded_right = [[] for _ in range(max_depth)]
                
                for i, (offset, llist, rlist) in enumerate(zip(batch_offset, left_raw, right_raw)):
                    for d in range(len(llist)):
                        l = np.where(llist[d] == -1, -1, llist[d] + offset).astype(np.int64)
                        r = np.where(rlist[d] == -1, -1, rlist[d] + offset).astype(np.int64)
                        padded_left[d].append(l)
                        padded_right[d].append(r)
                    for d in range(len(llist), max_depth):
                        padded_left[d].append(np.empty(0, dtype=np.int64))
                        padded_right[d].append(np.empty(0, dtype=np.int64))
                
                left_idx = [np.concatenate(padded_left[d]) for d in range(max_depth)] + [None]
                right_idx = [np.concatenate(padded_right[d]) for d in range(max_depth)] + [None]
            
            elif len(self.cached_indices) > 0:
                left_idx, right_idx = self.cached_indices[graph_ids[0]]
            
            else:
                left_idx, right_idx = self.GetMostRecentWeight(all_ids_len + 1, batch_last_edges=batch_last_edges)
                
            topdown_edge_index = (left_idx, right_idx)
        
        else:
            topdown_edge_index = None
        
        return topdown_edge_index
    
    def gen_row(self, ll, ll_wt, state, tree_node, col_sm, lb, ub, edge_feats=None, prev_state=None, num_nodes=None):
        assert lb <= ub
        if tree_node.is_root:
            if self.method != "BiGG-E":
                logit_has_edge = self.pred_has_ch(state[0])
                prob_has_edge = torch.sigmoid(logit_has_edge)
            else:
                logit_has_edge = self.get_merged_prob(state, prev_state, self.pred_has_ch, self.joint_merge_has_ch)
                prob_has_edge = torch.sigmoid(logit_has_edge)
            if col_sm.supervised:
                has_edge = len(col_sm.indices) > 0
            else:
                has_edge = np.random.rand() < self.get_prob_fix(prob_has_edge.item())
                if ub == 0:
                    has_edge = False
                if tree_node.n_cols <= 0:
                    has_edge = False
                if lb:
                    has_edge = True
            
            label = torch.tensor([has_edge], dtype=torch.float32, device=logit_has_edge.device)
            ll = ll - F.binary_cross_entropy_with_logits(logit_has_edge.view(-1, 1), label.view(-1, 1), reduction='sum')
            tree_node.has_edge = has_edge
        else:
            assert ub > 0
            tree_node.has_edge = True
        
        if not tree_node.has_edge:  # an empty tree
            return ll, ll_wt, self.get_empty_state(), 0, None, prev_state

        if tree_node.is_leaf:
            tree_node.bits_rep = [0]
            col_sm.add_edge(tree_node.col_range[0])
            
            if self.bits_compress > 0:
                if self.has_edge_feats:
                    assert self.method == "BiGG-E"
                    cur_feats = edge_feats[col_sm.pos - 1].unsqueeze(0) if col_sm.supervised else None
                    merged_state = self.get_merged_prob(state, prev_state, None, self.joint_update_wt)
                    edge_ll, cur_feats = self.predict_edge_feats(merged_state, cur_feats)
                    edge_embed = self.embed_edge_feats(cur_feats)
                    prev_state = (edge_embed[0].clone(), edge_embed[1].clone())
                    ll_wt = ll_wt + edge_ll
                    return ll, ll_wt, self.bit_rep_net(tree_node.bits_rep, tree_node.n_cols), 1, cur_feats, prev_state
                
                else:
                    return ll, ll_wt, self.bit_rep_net(tree_node.bits_rep, tree_node.n_cols), 1, None, None
            
            else:
                if self.has_edge_feats:
                    cur_feats = edge_feats[col_sm.pos - 1].unsqueeze(0) if col_sm.supervised else None
                    if self.method == "BiGG-E":
                        merged_state = self.get_merged_prob(state, prev_state, None, self.joint_update_wt)
                    
                    else:
                        merged_state = state
                    edge_ll, cur_feats = self.predict_edge_feats(merged_state, cur_feats)
                    edge_embed = self.embed_edge_feats(cur_feats)
                    ll_wt = ll_wt + edge_ll
                    if self.method == "BiGG-E":
                        prev_state = (edge_embed[0].clone(), edge_embed[1].clone())
                        return ll, ll_wt, (self.leaf_h0, self.leaf_c0), 1, cur_feats, prev_state
                    
                    else:
                        return ll, ll_wt, (edge_embed, edge_embed), 1, cur_feats, None
                
                else:
                    return ll, ll_wt, (self.leaf_h0, self.leaf_c0), 1, None, None
        else:
            tree_node.split()
            mid = (tree_node.col_range[0] + tree_node.col_range[1]) // 2
            if self.method != "BiGG-E":
                left_logit = self.pred_has_left(state[0][-1], lv = tree_node.depth)
                left_prob = left_prob = torch.sigmoid(left_logit)
            else:
                left_logit = self.get_merged_prob(state, prev_state, partial(self.pred_has_left, lv = tree_node.depth), self.joint_merge_lr)
                left_prob = torch.sigmoid(left_logit)
            
            if col_sm.supervised:
                has_left = col_sm.next_edge < mid
            else:
                has_left = np.random.rand() < self.get_prob_fix(left_prob.item())
                if ub == 0:
                    has_left = False
                if lb > tree_node.rch.n_cols:
                    has_left = True
            
            
            label = torch.tensor([has_left], dtype=torch.float32, device=left_logit.device)
            ll = ll - F.binary_cross_entropy_with_logits(left_logit.view(-1, 1), label.view(-1, 1), reduction='sum')
            
            left_pos = self.tree_pos_enc([tree_node.lch.n_cols])
            state = self.cell_topdown(self.topdown_left_embed[[int(has_left)]] + left_pos, state, tree_node.depth)
            pred_edge_feats = []
            if has_left:
                lub = min(tree_node.lch.n_cols, ub)
                llb = max(0, lb - tree_node.rch.n_cols)
                ll, ll_wt, left_state, num_left, left_edge_feats, prev_state = self.gen_row(ll, ll_wt, state, tree_node.lch, col_sm, llb, lub, edge_feats, prev_state=prev_state, num_nodes=num_nodes)
                pred_edge_feats.append(left_edge_feats)
            else:
                left_state = self.get_empty_state()
                num_left = 0

            right_pos = self.tree_pos_enc([tree_node.rch.n_cols])
            topdown_state = self.l2r_cell(state, (left_state[0] + right_pos, left_state[1] + right_pos), tree_node.depth)
            
            rlb = max(0, lb - num_left)
            rub = min(tree_node.rch.n_cols, ub - num_left)
            if not has_left:
                has_right = True
            else:
                if self.method != "BiGG-E":
                    right_logit = self.pred_has_right(topdown_state[0], lv = tree_node.depth)
                    right_prob = torch.sigmoid(right_logit)
                else:
                    right_logit = self.get_merged_prob(topdown_state, prev_state, partial(self.pred_has_right, lv = tree_node.depth), self.joint_merge_lr)
                    right_prob = torch.sigmoid(right_logit)
                
                
                if col_sm.supervised:
                    has_right = col_sm.has_edge(mid, tree_node.col_range[1])
                else:
                    has_right = np.random.rand() < self.get_prob_fix(right_prob.item())
                    if rub == 0:
                        has_right = False
                    if rlb:
                        has_right = True
                
                label = torch.tensor([has_right], dtype=torch.float32, device=right_logit.device)
                ll = ll - F.binary_cross_entropy_with_logits(right_logit.view(-1, 1), label.view(-1, 1), reduction='sum')
                #ll = ll + (torch.log(right_prob) if has_right else torch.log(1 - right_prob))
            
            topdown_state = self.cell_topright(self.topdown_right_embed[[int(has_right)]], topdown_state, tree_node.depth)

            if has_right:  # has edge in right child
                ll, ll_wt, right_state, num_right, right_edge_feats, prev_state = self.gen_row(ll, ll_wt, topdown_state, tree_node.rch, col_sm, rlb, rub, edge_feats, prev_state=prev_state, num_nodes=num_nodes)
                pred_edge_feats.append(right_edge_feats)
            else:
                right_state = self.get_empty_state()
                num_right = 0
            if tree_node.col_range[1] - tree_node.col_range[0] <= self.bits_compress:
                summary_state = self.bit_rep_net(tree_node.bits_rep, tree_node.n_cols)
            else:
                summary_state = self.lr2p_cell(left_state, right_state)
            if self.has_edge_feats:
                edge_feats = torch.cat(pred_edge_feats, dim=0)
            return ll, ll_wt, summary_state, num_left + num_right, edge_feats, prev_state
    
    def forward(self, node_end, edge_list=None, node_feats=None, edge_feats=None, node_start=0, list_states=[], lb_list=None, ub_list=None, col_range=None, num_nodes=None, display=False):
        pos = 0
        total_ll = 0.0
        total_ll_wt = 0.0
        edges = []
        if self.debug:
            row_states_h = []
            row_states_c = []
        self.num_edge = 0
        self.row_tree.reset(list_states)
        controller_state = self.row_tree()
        
        prev_state=None  
        if self.method == "BiGG-E":
            self.weight_tree.reset([])
            prev_state = self.weight_tree()
        
        if num_nodes is None:
            num_nodes = node_end
        pbar = range(node_start, node_end)
        if display:
            pbar = tqdm(pbar)
        list_pred_edge_feats = []
        for i in pbar:
            if edge_list is None:
                col_sm = ColAutomata(supervised=False)
            else:
                indices = []
                while pos < len(edge_list) and i == edge_list[pos][0]:
                    indices.append(edge_list[pos][1])
                    pos += 1
                indices.sort()
                col_sm = ColAutomata(supervised=True, indices=indices)

            cur_row = AdjRow(i, self.directed, self.self_loop, col_range=col_range)
            lb = 0 if lb_list is None else lb_list[i]
            ub = cur_row.root.n_cols if ub_list is None else ub_list[i]
            cur_pos_embed = self.row_tree.pos_enc([num_nodes - i])
            controller_state = [x + cur_pos_embed for x in controller_state]
            if self.has_edge_feats:
                target_edge_feats = None if edge_feats is None else edge_feats[len(edges) : len(edges) + len(col_sm)]
            else:
                target_edge_feats = None
            
            if self.debug:
                row_states_h.append(controller_state[0])
                row_states_c.append(controller_state[1])
            ll, ll_wt, cur_state, _, target_edge_feats, prev_state = self.gen_row(0, 0, controller_state, cur_row.root, col_sm, lb, ub, target_edge_feats, prev_state=prev_state, num_nodes=num_nodes)
            if target_edge_feats is not None and target_edge_feats.shape[0]:
                list_pred_edge_feats.append(target_edge_feats)
            
            assert lb <= len(col_sm.indices) <= ub
            controller_state = self.row_tree(cur_state)
            edges += [(i, x) for x in col_sm.indices]
            total_ll = total_ll + ll
            total_ll_wt = total_ll_wt + ll_wt
        
        if self.has_edge_feats:
            edge_feats = torch.cat(list_pred_edge_feats, dim=0)
            if self.sampling_method == "softplus":
                edge_feats = torch.nn.functional.softplus(edge_feats) 
            elif self.sampling_method == "lognormal":
                edge_feats = torch.exp(edge_feats)
            edge_feats = edge_feats / self.wt_scale
        
        if self.debug:
            row_states_h = torch.cat(row_states_h, dim = 0)
            row_states_c = torch.cat(row_states_c, dim = 0)
            row_states = (row_states_h, row_states_c)
            return total_ll, total_ll_wt, edges, self.row_tree.list_states, row_states, edge_feats
        
        return (total_ll, total_ll_wt), edges, self.row_tree.list_states, None, edge_feats



    def forward_row_trees(self, graph_ids, node_feats=None, edge_feats=None, list_node_starts=None, num_nodes=-1, list_col_ranges=None, batch_last_edges=None, list_num_edges=None):
        TreeLib.PrepareMiniBatch(graph_ids, list_node_starts, num_nodes, list_col_ranges)
        # embed trees
        all_ids = TreeLib.PrepareTreeEmbed()
        
        topdown_edge_index = self.get_topdown_idx(list_num_edges, batch_last_edges, graph_ids, len(all_ids) + 1)
        
        if self.bits_compress <= 0:
            h_bot = torch.cat([self.empty_h0, self.leaf_h0], dim=0)
            c_bot = torch.cat([self.empty_c0, self.leaf_c0], dim=0)
            fn_hc_bot = lambda d: (h_bot, c_bot)
        
        else:
            binary_embeds, base_feat = TreeLib.PrepareBinary()
            fn_hc_bot = lambda d: (binary_embeds[d], binary_embeds[d]) if d < len(binary_embeds) else base_feat
        
        max_level = len(all_ids) - 1
        h_buf_list = [None] * (len(all_ids) + 1)
        c_buf_list = [None] * (len(all_ids) + 1)
        
        for d in range(len(all_ids) - 1, -1, -1):
            fn_ids = lambda i: all_ids[d][i] #lambda i, d=d: all_ids[d][i]
            if d == max_level:
                h_buf = c_buf = None
            else:
                h_buf = h_buf_list[d + 1]
                c_buf = c_buf_list[d + 1]
            h_bot, c_bot = fn_hc_bot(d + 1)
            if self.method == "BiGG-MLP":
                edge_idx, is_rch = TreeLib.GetEdgeAndLR(d + 1)
                local_edge_feats = edge_feats[edge_idx] 
                new_h, new_c = featured_batch_tree_lstm2(local_edge_feats, is_rch, h_bot, c_bot, h_buf, c_buf, fn_ids, self.lr2p_cell)
            else:
                new_h, new_c = batch_tree_lstm2(h_bot, c_bot, h_buf, c_buf, fn_ids, self.lr2p_cell)
            h_buf_list[d] = new_h
            c_buf_list[d] = new_c
        hc_bot = fn_hc_bot(0)
        feat_dict = {}
        if self.method == "BiGG-MLP":
            edge_idx, is_rch = TreeLib.GetEdgeAndLR(0)
            local_edge_feats = edge_feats[edge_idx] 
            feat_dict['edge'] = (local_edge_feats, is_rch)
        if len(feat_dict):
            hc_bot = (hc_bot, feat_dict)
        return hc_bot, fn_hc_bot, h_buf_list, c_buf_list, topdown_edge_index

    def merge_states(self, update_idx, top_states, edge_feats_embed, func, predict_top=True):
        if predict_top:
            update_bool = (update_idx != -1).bool()
            cur_edge_idx = update_idx.clone()
            if not update_bool.all():
                last_col_idx = edge_feats_embed[0].shape[0] - 1
                cur_edge_idx = torch.where(update_bool, cur_edge_idx, last_col_idx)
            top_states_last = (top_states[0], top_states[1])
        
        else:
            update_bool = update_idx[0].bool()
            edge_of_lv = update_idx[1]
            cur_edge_idx = edge_of_lv - 1
            last_col_idx = edge_feats_embed[0].shape[0] - 1
            cur_edge_idx = torch.where(update_bool, cur_edge_idx, last_col_idx)
            
            if self.embed_dim != self.embed_dim_wt:
                top_states_last = (top_states[0], self.proj_top_c(top_states[1]))
            
            else:
                top_states_last = (top_states[0], top_states[1])
        
        # Select edge features
        edge_feats = (edge_feats_embed[0][cur_edge_idx], edge_feats_embed[1][cur_edge_idx])
        top_states_h, _ = func(top_states_last, edge_feats)
        return top_states_h, None

    def forward_train(self, graph_ids, node_feats=None,
                      list_node_starts=None, num_nodes=-1, prev_rowsum_states=[None, None], list_col_ranges=None, edge_feat_info=None):
        ll = 0.0
        ll_wt = 0.0
        noise = 0.0
        
        ## Extract Edge Feat Info (if any)
        edge_feats = edge_feat_info['edge_feats']
        list_num_edges = edge_feat_info['batch_num_edges']
        batch_last_edges = edge_feat_info['batch_last_edges']
        lv_lists = edge_feat_info['cur_lv_lists']
        first_edge = edge_feat_info['first_edge']
        
        ## Embed Edge feats and Compute Row States
        edge_feats_embed = self.embed_edge_feats(edge_feats, list_num_edges=list_num_edges, lv_lists=lv_lists) if self.has_edge_feats else None
        if self.method == "BiGG-E":
            edge_feats_embed_wt = (torch.cat([edge_feats_embed[0], self.weight_tree.init_h0_wt], dim=0),
                                torch.cat([edge_feats_embed[1], self.weight_tree.init_c0_wt], dim=0))
            
            if self.embed_dim_wt != self.embed_dim:
                edge_feats_embed = (edge_feats_embed_wt[0], self.proj_wt_c(edge_feats_embed_wt[1]))
        hc_bot, fn_hc_bot, h_buf_list, c_buf_list, topdown_edge_index = self.forward_row_trees(graph_ids, node_feats, edge_feats_embed if self.method == "BiGG-MLP" else None, list_node_starts, num_nodes, list_col_ranges, batch_last_edges, list_num_edges)
        row_states, next_states = self.row_tree.forward_train(*hc_bot, h_buf_list[0], c_buf_list[0], *prev_rowsum_states)
        if self.debug:
            ret_h = row_states[0].clone()
            ret_c = row_states[1].clone()
            ret = (ret_h, ret_c)
        
        ## Predict Each Row Having an Edge
        if self.method == "BiGG-E":
            batch_last_edges = torch.from_numpy(batch_last_edges).to(self.device)
            row_states_wt = self.merge_states(batch_last_edges, row_states, edge_feats_embed, func = self.joint_merge_has_ch)
            logit_has_edge = self.pred_has_ch(row_states_wt[0])
        
        else:
            logit_has_edge = self.pred_has_ch(row_states[0])
                
        has_ch, _ = TreeLib.GetChLabel(0, dtype=bool)
        ll = ll + self.binary_ll(logit_has_edge, has_ch)
        cur_states = (row_states[0][has_ch], row_states[1][has_ch])
                    
        lv=0
        while True:
            is_nonleaf = TreeLib.QueryNonLeaf(lv)
            if self.has_edge_feats:
                edge_of_lv = TreeLib.GetEdgeOf(lv)
                edge_state = (cur_states[0][~is_nonleaf], cur_states[1][~is_nonleaf])
                target_feats = edge_feats[edge_of_lv]
                if self.method == "BiGG-E": 
                    if edge_of_lv is None:
                        edge_of_lv = np.array([])
                    has_prev = []
                    has_prev_np = ~np.isin(edge_of_lv, first_edge)
                    has_prev = torch.from_numpy(has_prev_np).to(self.device)
                    edge_of_lv = torch.from_numpy(edge_of_lv).to(self.device)
                    edge_state_wt = self.merge_states([has_prev, edge_of_lv], edge_state, edge_feats_embed_wt, func = self.joint_update_wt, predict_top = False)
                    edge_ll, _ = self.predict_edge_feats(edge_state_wt, target_feats)
                
                else:
                    edge_ll, _ = self.predict_edge_feats(edge_state, target_feats)
                ll_wt = ll_wt + edge_ll
            
            if is_nonleaf is None or np.sum(is_nonleaf) == 0:
                break
            cur_states = (cur_states[0][is_nonleaf], cur_states[1][is_nonleaf])     
            
            if self.method == "BiGG-E":
                cur_left_updates = topdown_edge_index[0][lv]
                cur_left_updates = torch.from_numpy(cur_left_updates).to(self.device)
                cur_states_wt = self.merge_states(cur_left_updates, cur_states, edge_feats_embed, func = self.joint_merge_lr)
                left_logits = self.pred_has_left(cur_states_wt[0], lv)
            else:
                left_logits = self.pred_has_left(cur_states[0], lv)
            
            has_left, num_left = TreeLib.GetChLabel(-1, lv)
            left_update = self.topdown_left_embed[has_left] + self.tree_pos_enc(num_left)
            left_ll, float_has_left = self.binary_ll(left_logits, has_left, need_label=True, reduction='sum')
            ll = ll + left_ll

            cur_states = self.cell_topdown(left_update, cur_states, lv)

            left_ids = TreeLib.GetLeftRootStates(lv)
            h_bot, c_bot = fn_hc_bot(lv + 1)
            
            if lv + 1 < len(h_buf_list):
                h_next_buf, c_next_buf = h_buf_list[lv + 1], c_buf_list[lv + 1]
            else:
                h_next_buf = c_next_buf = None

            if self.method == "BiGG-MLP":
                h_bot, c_bot = h_bot[left_ids[0]], c_bot[left_ids[0]]
                edge_idx, is_rch = TreeLib.GetEdgeAndLR(lv + 1)
                left_feats = edge_feats_embed[edge_idx[~is_rch]] 
                h_bot, c_bot = selective_update_hc(h_bot, c_bot, left_ids[0], left_feats)
                left_ids = tuple([None] + list(left_ids[1:]))
            
            left_subtree_states = tree_state_select(h_bot, c_bot,
                                                    h_next_buf, c_next_buf,
                                                    lambda: left_ids)
            
            has_right, num_right = TreeLib.GetChLabel(1, lv)
            right_pos = self.tree_pos_enc(num_right)
            left_subtree_states = [x + right_pos for x in left_subtree_states]
            topdown_state = self.l2r_cell(cur_states, left_subtree_states, lv)
            
            if self.method == "BiGG-E":
                cur_right_updates = topdown_edge_index[1][lv]
                cur_right_updates = torch.from_numpy(cur_right_updates).to(self.device)
                topdown_wt_state = self.merge_states(cur_right_updates, topdown_state, edge_feats_embed, func = self.joint_merge_lr)
                right_logits = self.pred_has_right(topdown_wt_state[0], lv)
            
            else:
                right_logits = self.pred_has_right(topdown_state[0], lv)
            
            right_update = self.topdown_right_embed[has_right]
            topdown_state = self.cell_topright(right_update, topdown_state, lv)
            right_ll = self.binary_ll(right_logits, has_right, reduction='none') * float_has_left
            ll = ll + torch.sum(right_ll)
            
            lr_ids = TreeLib.GetLeftRightSelect(lv, np.sum(has_left), np.sum(has_right))
            new_states = []
            for i in range(2):
                new_s = multi_index_select([lr_ids[0], lr_ids[2]], [lr_ids[1], lr_ids[3]],
                                            cur_states[i], topdown_state[i])
                new_states.append(new_s)
            cur_states = tuple(new_states)
            lv += 1
        
        if self.debug:
            return ll, ll_wt, ret
        return ll, ll_wt, next_states













































