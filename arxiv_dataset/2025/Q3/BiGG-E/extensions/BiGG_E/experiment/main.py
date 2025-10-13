from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import cmd
from copyreg import pickle
# pylint: skip-file

import os
import sys
import numpy as np
import pickle as cp
import networkx as nx
import random
from tqdm import tqdm
import torch
import torch.optim as optim
from collections import OrderedDict
from torch.nn import functional as F
from bigg.experiments.train_utils import get_node_dist
from bigg.model.tree_clib.tree_lib import setup_treelib, TreeLib
from extensions.BiGG_E.model_extensions.customized_models import BiGGExtension, BiGGWithGCN
from extensions.BiGG_E.model_extensions.util_extension.train_util import *
from extensions.evaluate.graph_stats import *
from extensions.evaluate.mmd import *
from extensions.evaluate.mmd_stats import *
from extensions.common.configs import cmd_args, set_device


if __name__ == '__main__':
    # == TreeLib Setup ==
    random.seed(cmd_args.seed)
    torch.manual_seed(cmd_args.seed)
    np.random.seed(cmd_args.seed)
    set_device(cmd_args.gpu)
    if cmd_args.method == 'BiGG-MLP':
        cmd_args.bits_compress = 0 # disable bits compression for BiGG-MLP
    if not cmd_args.has_edge_feats:
        cmd_args.method = "None"
    setup_treelib(cmd_args)
    assert cmd_args.blksize < 0  # assume graph is not that large, otherwise model parallelism is needed    
    
    # == Load training graphs ==
    train_path = os.path.join(cmd_args.data_dir, 'train-graphs.pkl')
    with open(train_path, 'rb') as f:
        train_graphs = cp.load(f)
    cmd_args.max_num_nodes = max([len(gg.nodes) for gg in train_graphs])

    # == Tuning and debug ==
    if cmd_args.tune_sigma:
        tune_sigma(train_graphs, cmd_args.g_type) # Tuning sigma for Tree Weight MMD
    
    elif cmd_args.debug:
        batch_indices = set_debug_args(cmd_args)
    
    # == Process train graphs ==
    if cmd_args.phase == "train":
        for g in train_graphs:
            TreeLib.InsertGraph(g)
        list_edge_feats, list_num_edges, lv_lists, list_last_edges = process_edge_feats(cmd_args, train_graphs)
        print('# graphs', len(train_graphs), 'max # nodes', cmd_args.max_num_nodes)
    
    # == Build and load model and optimizers ==
    model, optimizer_topo, optimizer_wt, params_list = build_model_and_optimizers_main(cmd_args) 
    
    ######################################################################################
    if cmd_args.phase in ['val', 'test']:
        path = os.path.join(cmd_args.data_dir, f'{cmd_args.phase}-graphs.pkl')
        with open(path, 'rb') as f:
            gt_graphs = cp.load(f)
        if cmd_args.num_test_gen == -1:
            cmd_args.num_test_gen = len(gt_graphs)
        
        print("======================================================")
        print("Now performing phase: ", cmd_args.phase)
        print("Model Loaded using Method: ", cmd_args.method)
        print("Has Edge Feats? ", bool(cmd_args.has_edge_feats))
        print("Number of Graphs to Generate: ", cmd_args.num_test_gen)
        print('# ground truth graphs', len(gt_graphs))
        print("======================================================")
        
        with torch.no_grad():
            print("Starting graph generation...")
            model.eval()
            gen_graphs = []
            
            # == Get number of nodes for val / test graphs
            num_node_dist = get_node_dist(train_graphs)
            num_nodes_list = get_num_nodes(train_graphs, gt_graphs, num_node_dist, cmd_args)
            print("Num node list: ", num_nodes_list)
            
            # == Sample
            for idx in tqdm(range(cmd_args.num_test_gen)):
                num_nodes = num_nodes_list[idx]
                _, pred_edges, _, _, pred_edge_feats = model(node_end = num_nodes, display=cmd_args.display)
                if cmd_args.method == 'BiGG-GCN':
                    fix_edges = [(min(e1, e2), max(e1, e2)) for e1, e2 in pred_edges]
                    pred_edge_tensor = torch.tensor(fix_edges).to(cmd_args.device)
                    pred_weighted_tensor = model.gcn_mod.sample(num_nodes, pred_edge_tensor).cpu().numpy()
                    weighted_edges = [(int(e1), int(e2), round(w.item(), 4)) for e1, e2, w in pred_weighted_tensor]
                
                elif cmd_args.has_edge_feats:
                    weighted_edges = [(min(e[0], e[1]), max(e[0], e[1]), round(w.item(), 4)) for e, w in zip(pred_edges, pred_edge_feats)]
                    
                else:
                    weighted_edges = [(min(e[0], e[1]), max(e[0], e[1]), 1.0) for e in pred_edges]
                
                pred_g = nx.Graph()
                pred_g.add_weighted_edges_from(weighted_edges)
                gen_graphs.append(pred_g)
        
        print('Graph generation complete')
        
        if cmd_args.g_type == "tree" and cmd_args.phase == "test":
            print('Saving Graph Weight and Summary Stats')
            gen_stats = extract_weights_with_stats(gen_graphs)            
            with open(cmd_args.model_dump + '.gen_stats', 'wb') as f:
                cp.dump(gen_stats, f, cp.HIGHEST_PROTOCOL)
            
            if cmd_args.method == "BiGG-E":
                gt_stats = extract_weights_with_stats(gt_graphs)
                with open(cmd_args.model_dump + '.gt_stats', 'wb') as f:
                    cp.dump(gt_stats, f, cp.HIGHEST_PROTOCOL)
        
        if cmd_args.method == "BiGG-E" and cmd_args.phase == "test":
            print('saving BiGG-E graphs')
            with open(cmd_args.model_dump + '.graphs', 'wb') as f:
                cp.dump(gen_graphs, f, cp.HIGHEST_PROTOCOL)
        
        print("Generating Ground Truth Graph Stats")
        get_graph_stats(gen_graphs, gt_graphs, cmd_args.g_type)
        
        sys.exit()
    ######################################################################################
    model.train()
    
    if cmd_args.method == "BiGG-E":
        print("Caching LR Indices")
        model.CacheTopdownIdx(train_graphs, list_last_edges)
    
    if cmd_args.debug:
        debug_model(model, train_graphs, list_edge_feats, list_last_edges, batch_indices, lv_lists, cmd_args)
    
    # === Setup Phase Parameters ===
    lr_scheduler = {'lobster': 100, 'tree': 200, 'db': 2000, 'er': 250, 'span': 500, 'franken': 200}
    epoch_lr_decrease = cmd_args.epoch_plateu if cmd_args.epoch_plateu > -1 else lr_scheduler[cmd_args.g_type]
    top_lr_decrease = cmd_args.top_plateu if cmd_args.top_plateu > -1 else lr_scheduler[cmd_args.g_type]
    wt_lr_decrease = cmd_args.wt_plateu if cmd_args.top_plateu > -1 else lr_scheduler[cmd_args.g_type]
    
    # == Num nodes ==
    num_nodes_list = [len(g) for g in train_graphs]
    
    # == Epoch / Iteration Counters ==
    grad_accum_counter = 0
    N, B = len(train_graphs), cmd_args.batch_size
    num_iter = N // B
    num_steps = N // (cmd_args.accum_grad * B)
    cmd_args.epoch_load = cmd_args.epoch_load or 0
    indices = list(range(N))
    
    # === Logging and Initialization ===
    loss_tops_list = np.zeros(cmd_args.num_epochs - cmd_args.epoch_load)
    loss_wts_list = np.zeros_like(loss_tops_list)
    plateu_wt_loss = False
    
    # === Training Setup ===
    print("Begining Training")
    print("Current Topology Learning Rate is:", cmd_args.learning_rate_top)
    print("Current Weight Learning Rate is:", cmd_args.learning_rate_wt)
    print("Dividing Weight Loss by:", cmd_args.scale_loss)
    print("Starting at epoch: ", cmd_args.epoch_load)
    print("Total number of epochs: ", cmd_args.num_epochs)
    print("Reduce Topology LR at epoch # : ", top_lr_decrease)
    print("Reduce Weight LR at epoch # : ", wt_lr_decrease)
    
    # Update weight stats at start of training
    if cmd_args.has_edge_feats and cmd_args.epoch_load == 0:
        print("Updating Weight Stats")
        update_weight_stats(model, list_edge_feats, cmd_args.method)
    
    for epoch in range(cmd_args.epoch_load, cmd_args.num_epochs):
        optimizer_topo.zero_grad()
        if cmd_args.has_edge_feats:
            optimizer_wt.zero_grad()
        
        epoch_loss_top, epoch_loss_wt = 0.0, 0.0
        grad_accum_counter = 0
        pbar = tqdm(range(num_iter)) 
        random.shuffle(indices)
        
        batch_dict = precompute_batch_indices(indices, list_num_edges, list_num_edges, cmd_args)
        
        # == Check Plateu ==
        cur = epoch + 1
        if cur == top_lr_decrease:
            print("Lowering Topology Learning Rate to: ", 1e-5)
            cmd_args.learning_rate_top = 1e-5
            optimizer_topo = optim.AdamW(params_list[0], lr=cmd_args.learning_rate_top , weight_decay=1e-4)
        
        if cmd_args.has_edge_feats and cur == wt_lr_decrease:
            print("Lowering Weight Learning Rate to: ", 1e-5)
            cmd_args.learning_rate_wt = 1e-5
            decay_wt = 1e-4
            if cmd_args.method != "BiGG-GCN":
                cmd_args.scale_loss = 100
                decay_wt = 1e-3
            optimizer_wt = optim.AdamW(params_list[1], lr=cmd_args.learning_rate_wt, weight_decay=decay_wt)
        
        for idx in pbar:
            if num_iter - idx < cmd_args.accum_grad - grad_accum_counter:
                print("Skipping iteration: ", idx + 1)
                continue
            
            start = batch_dict[idx]['start']
            stop = batch_dict[idx]['stop']
            batch_indices = batch_dict[idx]['batch_indices']
            cur_num_nodes = batch_dict[idx]['cur_num_nodes']
            cur_num_edges = batch_dict[idx]['cur_num_edges']
            num_nodes = batch_dict[idx]['num_nodes']
            num_edges = batch_dict[idx]['num_edges']
            
            # === Forward Pass ===
            if cmd_args.method == 'BiGG-GCN':
                feat_idx, edge_list, batch_weight_idx = GCNN_batch_train_graphs(train_graphs, batch_indices, cmd_args)
                edge_feat_info = batch_edge_info(batch_indices) ## Retrieves null list for unweighted BiGG
                ll, ll_wt = model.forward_train2(batch_indices, feat_idx, edge_list, batch_weight_idx, edge_feat_info)
            
            else:
                edge_feat_info = batch_edge_info(batch_indices, list_edge_feats, lv_lists, list_last_edges, list_num_edges)
                ll, ll_wt, _ = model.forward_train(batch_indices, edge_feat_info = edge_feat_info)
            
            # === Model Loss Calculation ===
            if cmd_args.g_type == "db":
                weight_factor = ((epoch % 5 == 0) * 1 + 1e-5 * (epoch % 5!= 0) if cmd_args.learning_rate_top == 1e-5 and cmd_args.learning_rate_wt == 1e-5 else 1.0)
            else:
                weight_factor = (epoch % 2 * 1 + 1e-5 * (epoch % 2) if cmd_args.learning_rate_top == 1e-5 and cmd_args.learning_rate_wt == 1e-5 else 1.0)
            weight_factor = (1.0 if cmd_args.method == "BiGG-GCN" else weight_factor)
            loss = -ll / num_nodes  - weight_factor * ll_wt / (cmd_args.scale_loss * num_edges)
            
            # === Backward and Optimizer Step ===
            loss.backward()
            grad_accum_counter += 1
            if grad_accum_counter == cmd_args.accum_grad:
                param_step(model, params_list, cmd_args, optimizer_topo, optimizer_wt)
                grad_accum_counter = 0
            
            # === Loss Calculations ===
            true_loss = -ll / cur_num_nodes - ll_wt / cur_num_edges
            epoch_loss_top = (-ll / (num_nodes * num_steps)).item() + epoch_loss_top
            if cmd_args.has_edge_feats:
                epoch_loss_wt = (-ll_wt / (num_edges * num_steps)).item() + epoch_loss_wt
            pbar.set_description('epoch %.2f, loss: %.4f' % (epoch + (idx + 1) / num_iter, true_loss))
        
        # == Update Epoch Avg. Losses + Summary ==
        loss_tops_list[epoch - cmd_args.epoch_load] = epoch_loss_top
        loss_wts_list[epoch - cmd_args.epoch_load] = epoch_loss_wt 
        
        print('epoch complete')
        print("Epoch Loss (Topology): ", epoch_loss_top)
        print("Epoch Loss (Weights): ", epoch_loss_wt)
        
        # === Save Checkpoint ===
        cur = epoch + 1
        if cur % cmd_args.epoch_save == 0 or cur == cmd_args.num_epochs:
            print('saving epoch')
            checkpoint = {'epoch': epoch, 
                          'model': model.state_dict(), 
                          'optimizer_topo': optimizer_topo.state_dict(), 
                          'optimizer_wt': (optimizer_wt.state_dict() if optimizer_wt is not None else None), 
                          'learning_rate_top': cmd_args.learning_rate_top,
                          'learning_rate_wt': cmd_args.learning_rate_wt}
            torch.save(checkpoint, os.path.join(cmd_args.save_dir, 'epoch-%d.ckpt' % cur))
        
    # === Wrap Up ===
    if len(loss_tops_list) > 0:
        path = os.path.join(cmd_args.save_dir, 'loss_data.pkl')
        loss_data = {'loss_tops': loss_tops_list, 'loss_wts': loss_wts_list}
        torch.save(loss_data, path)
    
    print("Training Complete")
    sys.exit()    
        













































































