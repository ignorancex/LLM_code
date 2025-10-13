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
import gc
import torch
import torch.optim as optim
from datetime import datetime
from collections import OrderedDict
from extensions.BiGG_E.model_extensions.customized_models import BiGGExtension, BiGGWithGCN
from extensions.BiGG_E.model_extensions.util_extension.train_util import *
from bigg.model.tree_clib.tree_lib import setup_treelib, TreeLib
from extensions.evaluate.graph_stats import *
from extensions.evaluate.mmd import *
from extensions.evaluate.mmd_stats import *
from extensions.common.configs import cmd_args, set_device
from extensions.synthetic_gen.data_util import *
from extensions.synthetic_gen.data_creator import *
from bigg.experiments.train_utils import get_node_dist


if __name__ == '__main__': 
    if cmd_args.batch_size * cmd_args.accum_grad == 20:
        cmd_args.learning_rate_top = 5e-4
        cmd_args.learning_rate_wt = 5e-4 
    if cmd_args.method == "BiGG-MLP":
        cmd_args.bits_compress = 0
    elif cmd_args.method == "BiGG-GCN" and cmd_args.num_leaves != 7500:
        cmd_args.scale_loss = 1.0
    random.seed(cmd_args.seed)
    torch.manual_seed(cmd_args.seed)
    np.random.seed(cmd_args.seed)
    set_device(cmd_args.gpu)
    cmd_args.has_edge_feats = False
    setup_treelib(cmd_args)
    cmd_args.has_edge_feats = True
    
    print("~~~~~~~~~~~~~~~")
    print("Graph Type: ", cmd_args.g_type)
    print("Ordering: ", cmd_args.node_order)
    print("Number of Leaves: ", cmd_args.num_leaves)
    print("~~~~~~~~~~~~~~")
    
    if cmd_args.debug:
        batch_indices = set_debug_args(cmd_args)
    
    if cmd_args.phase == "train" and not cmd_args.training_time:
        cmd_args.epoch_load = cmd_args.epoch_load or 0
        cmd_args.num_graphs = 100
        path = os.path.join(cmd_args.base_path, cmd_args.method, cmd_args.g_type, 'graphs', f'temp-{2 * cmd_args.num_leaves}-graphs.ckpt')
        if os.path.isfile(path):
            print("Loading Saved Graphs")
            with open(path, 'rb') as f:
                graphs = cp.load(f)
        
        elif cmd_args.g_type in ["joint_2", "tree"]:
            graphs = create_training_graphs(cmd_args)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'wb') as f:
               cp.dump(graphs, f)
        
        else:
            print("Scalability run only uses tree graphs")
            sys.exit()
        
        # Get Training and Test Graphs
        train_graphs, test_graphs = process_graphs(graphs, cmd_args.node_order)
        cmd_args.max_num_nodes = max([len(gg.nodes) for gg in train_graphs])
        mean_num_nodes = np.mean([len(gg.nodes) for gg in train_graphs])
        [TreeLib.InsertGraph(g) for g in train_graphs]
        list_edge_feats, list_num_edges, lv_lists, list_last_edges = process_edge_feats(cmd_args, train_graphs)
        
        print('# graphs: ', len(train_graphs), '| max # nodes: ', cmd_args.max_num_nodes, '| mean # nodes: ', mean_num_nodes)
    
    ######################################################################################
    if cmd_args.phase != "train": 
        cmd_args.epoch_load = -1
        
        samp_time = time_model_forward(cmd_args.method, cmd_args, num_nodes = 2  * cmd_args.num_leaves - 1)
        print("Model: ", cmd_args.method)
        print("Number of Leaves: ", cmd_args.num_leaves)
        print(f"Time to sample 1 graph: {samp_time:.4f} seconds")
        sys.exit()
    #####################################################################################
    
    #####################################################################################
    if cmd_args.training_time:
        print("Getting training times")
        bigg_mlp_times, bigg_e_times, gcn_times = [], [], []
        cmd_args.num_graphs = 1
        random.seed(cmd_args.seed)
        torch.manual_seed(cmd_args.seed)
        np.random.seed(cmd_args.seed)

        num_leaves = cmd_args.num_leaves
        print("Number of Leaves: ", num_leaves)
        print("Method: ", cmd_args.method)
        
        num_nodes = 2 * int(num_leaves) - 1
        m = num_nodes - 1
        cmd_args.num_leaves = num_leaves
        g = graph_generator(cmd_args)
        g = get_graph_data(g[0], 'BFS', global_source=0)
        g = g[0]
        
        [TreeLib.InsertGraph(g)]
        edge_feats = torch.from_numpy(get_edge_feats(g)).to(cmd_args.device)
        edge_feat_info = batch_edge_info([0])
        cmd_args.max_num_nodes = num_nodes
        memories = []
        times =[]
        
        if cmd_args.method == "BiGG-GCN":
            feat_idx, edge_list, batch_weight_idx = GCNN_batch_train_graphs([g], [0], cmd_args)
        
        elif cmd_args.method == "BiGG-E":
            ## List num edges
            list_num_edges = [m]
            
            ## list_last_edge
            batch_last_edges = np.array(get_last_edge(g))
            cur_lv_lists = [get_single_lv_list(m)]
            edge_feat_info = {'edge_feats': edge_feats, 'batch_last_edges': batch_last_edges, 'cur_lv_lists': cur_lv_lists, 'batch_num_edges': list_num_edges, 'first_edge': [0]}
            
        
        elif cmd_args.method == "BiGG-MLP":
            ### BiGG-MLP
            cmd_args.bits_compress = 0
            edge_feat_info = {'edge_feats': edge_feats, 'batch_last_edges': None, 'cur_lv_lists': None, 'batch_num_edges': None, 'first_edge': None}
        
        for i in range(10):
            if cmd_args.method == "BiGG-GCN":
                cmd_args.has_edge_feats = False
                cmd_args.has_node_feats = False
                model = BiGGWithGCN(cmd_args).to(cmd_args.device)
                cmd_args.has_edge_feats = True
                optimizer = optim.AdamW(model.parameters(), lr=cmd_args.learning_rate, weight_decay=1e-4)
                
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                init_gcn = datetime.now()
                ll, ll_wt = model.forward_train2([0], feat_idx, edge_list, batch_weight_idx, edge_feat_info)
                loss = -ll / num_nodes - ll_wt / m
                loss.backward()    
                optimizer.step()
                optimizer.zero_grad()
                torch.cuda.synchronize()
                cur_gcn = datetime.now() - init_gcn
                
                mem_used = torch.cuda.max_memory_allocated() / 1024**2 
                
                if i >= 3:
                    times.append(cur_gcn.total_seconds())
                    memories.append(mem_used)
            
            elif cmd_args.method == "BiGG-E":
                model = BiGGExtension(cmd_args).to(cmd_args.device)
                optimizer = optim.AdamW(model.parameters(), lr=cmd_args.learning_rate, weight_decay=1e-4)
                model.update_weight_stats(edge_feats)
                model.CacheTopdownIdx([g], [batch_last_edges])
                
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                init_e = datetime.now()
                ll, ll_wt, _ = model.forward_train([0], edge_feat_info = edge_feat_info)
                
                loss = -ll / num_nodes - ll_wt / m
                loss.backward()    
                optimizer.step()
                optimizer.zero_grad()
                torch.cuda.synchronize()
                cur_e = datetime.now() - init_e
                
                mem_used = torch.cuda.max_memory_allocated() / 1024**2 
            
                if i >= 3:
                    times.append(cur_e.total_seconds())
                    memories.append(mem_used)
                
                
            elif cmd_args.method == "BiGG-MLP":
                model = BiGGExtension(cmd_args).to(cmd_args.device)
                optimizer = optim.AdamW(model.parameters(), lr=cmd_args.learning_rate, weight_decay=1e-4)
                model.update_weight_stats(edge_feats)
                
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                init_mlp = datetime.now()
                ll, ll_wt, _ = model.forward_train([0], edge_feat_info = edge_feat_info)
                
                loss = -ll / num_nodes - ll_wt / m
                loss.backward()    
                optimizer.step()
                optimizer.zero_grad()
                torch.cuda.synchronize()
                cur_mlp = datetime.now() - init_mlp
                
                mem_used = torch.cuda.max_memory_allocated() / 1024**2 
                
                if i >= 3:
                    times.append(cur_mlp.total_seconds())
                    memories.append(mem_used)
            
            del model
            del optimizer
            torch.cuda.empty_cache()
            gc.collect()
        print("Times: ", times)
        print("Average Time in 7 runs: ", np.mean(times))
        print("Average Peak Memory in 7 runs: ", np.mean(memories))
        sys.exit()
    #####################################################################################
    
    ### == Initialize Model; Prep Training ==
    model, optimizer_topo, optimizer_wt, params_list = build_model_and_optimizers(cmd_args) 
    model.train()
    cmd_args.num_epochs = max(cmd_args.num_epochs, cmd_args.epoch_load)
    
    ## quick test
    if cmd_args.method == "BiGG-E" and (cmd_args.epoch_load < cmd_args.num_epochs or cmd_args.debug):
        print("Caching LR Indices")
        model.CacheTopdownIdx(train_graphs, list_last_edges)
    if cmd_args.debug:
        debug_model(model, train_graphs, list_edge_feats, list_last_edges, batch_indices, lv_lists, cmd_args)
    
    # == Get Training Hyperparameters == 
    top_lr_decrease = cmd_args.top_plateu 
    wt_lr_decrease = cmd_args.wt_plateu
    
    grad_accum_counter = 0
    N, B = len(train_graphs), cmd_args.batch_size
    num_iter = N // B
    num_steps = N // (cmd_args.accum_grad * B)
    indices = list(range(N))
    
    # === Logging Losses ===
    loss_tops_list = np.zeros(cmd_args.num_epochs - cmd_args.epoch_load)
    loss_wts_list = np.zeros_like(loss_tops_list)

    # == Set Plateu ==
    if cmd_args.epoch_plateu < 0:
        cmd_args.epoch_plateu = cmd_args.num_epochs
    
    top_lr_decrease = cmd_args.epoch_plateu
    wt_lr_decrease = cmd_args.epoch_plateu
    
    # == Num nodes ==
    num_nodes_list = [len(g) for g in train_graphs]

    # === Training Setup ===
    print("====================================")
    print("Beginning Training")
    print("Method: ", cmd_args.method)
    print("Number of leaves: ", cmd_args.num_leaves)
    print("Current Topology Learning Rate is:", cmd_args.learning_rate_top)
    print("Current Weight Learning Rate is:", cmd_args.learning_rate_wt)
    print("Dividing Weight Loss by:", cmd_args.scale_loss)
    print("Starting at epoch: ", cmd_args.epoch_load)
    print("Total number of epochs: ", cmd_args.num_epochs)
    print("Reduce Topology LR at epoch # : ", top_lr_decrease)
    print("Reduce Weight LR at epoch # : ", wt_lr_decrease)
    print("====================================")
    
    # == Update stats at start of training ==
    if cmd_args.epoch_load == 0:
        for i, edge_feats in enumerate(list_edge_feats):
            initialize=(i+1 == len(list_edge_feats))
            if cmd_args.method == "BiGG-GCN":
                model.gcn_mod.update_weight_stats(edge_feats)
            else:
                model.update_weight_stats(edge_feats, initialize)
    
    for epoch in range(cmd_args.epoch_load, cmd_args.num_epochs):
        ## Zero out optimizers
        optimizer_topo.zero_grad()
        optimizer_wt.zero_grad()
        
        ## Epoch initialization
        epoch_loss_top, epoch_loss_wt = 0.0, 0.0
        grad_accum_counter = 0
        pbar = tqdm(range(num_iter)) 
        random.shuffle(indices)
        
        batch_dict = precompute_batch_indices(indices, num_nodes_list, list_num_edges, cmd_args)
        
        # == Check Plateu ==
        if epoch == top_lr_decrease:
            print("Lowering Topology Learning Rate to: ", 1e-5)
            cmd_args.learning_rate_top = 1e-5
            optimizer_topo = optim.AdamW(params_list[0], lr=cmd_args.learning_rate_top , weight_decay=1e-4)
        
        if epoch == wt_lr_decrease:
            print("Lowering Weight Learning Rate to: ", 1e-5)
            cmd_args.learning_rate_wt = 1e-5 
            decay_wt = 1e-4
            if cmd_args.method != "BiGG_GCN":
                cmd_args.scale_loss = 100
                decay_wt = 1e-3
            optimizer_wt = optim.AdamW(params_list[1], lr=cmd_args.learning_rate_wt, weight_decay=decay_wt)
        
        for idx in pbar:
            # === Get Batch Indices ===
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
                edge_feat_info = batch_edge_info(batch_indices) 
                ll, ll_wt = model.forward_train2(batch_indices, feat_idx, edge_list, batch_weight_idx, edge_feat_info)
            
            else:
                edge_feat_info = batch_edge_info(batch_indices, list_edge_feats, lv_lists, list_last_edges, list_num_edges)
                ll, ll_wt, _ = model.forward_train(batch_indices, edge_feat_info = edge_feat_info)
            
            # === Model Loss Computation ===
            weight_factor = (epoch % 2 * 1 + 1e-5 * (epoch % 2) if cmd_args.learning_rate_top == 1e-5 and cmd_args.learning_rate_wt == 1e-5 else 1.0)
            weight_factor = (1.0 if cmd_args.method == "BiGG-GCN" or cmd_args.g_type == "joint_2" else weight_factor)
            
            if cmd_args.num_leaves != 7500:
                loss = -ll / num_nodes  - weight_factor * ll_wt / (cmd_args.scale_loss * num_edges)
            
            else:
                loss = -ll / cur_num_nodes  - weight_factor * ll_wt / (cmd_args.scale_loss * cur_num_edges)
            # === Backward and Optimizer Step ===
            loss.backward()
            grad_accum_counter += 1
            if grad_accum_counter == cmd_args.accum_grad:
                if cmd_args.num_leaves == 7500:
                    for param in model.parameters():
                        if param.grad is not None:
                            param.grad.data.div_(cmd_args.accum_grad)
                param_step(model, params_list, cmd_args, optimizer_topo, optimizer_wt)
                grad_accum_counter = 0
            
            # === Loss Calculations ===
            true_loss = -ll / cur_num_nodes - ll_wt / cur_num_edges
            epoch_loss_top = (-ll / (num_nodes * num_steps)).item() + epoch_loss_top
            epoch_loss_wt = (-ll_wt / (num_edges * num_steps)).item() + epoch_loss_wt
            pbar.set_description('epoch %.2f, loss: %.4f' % (epoch + (idx + 1) / num_iter, true_loss))
        
        # == Update Epoch Avg. Losses + Summary ==
        loss_tops_list[epoch - cmd_args.epoch_load] = epoch_loss_top
        loss_wts_list[epoch - cmd_args.epoch_load] = epoch_loss_wt 
        print('epoch complete')
        print("Epoch Loss (Topology): ", epoch_loss_top)
        print("Epoch Loss (Weights): ", epoch_loss_wt)
    
        # == Check Save ==
        if (epoch+1) % 20 == 0:
            save_model(epoch, model, optimizer_topo, optimizer_wt, cmd_args, epoch_loss_top)
    
    if len(loss_tops_list) > 0:
        path = os.path.join(cmd_args.base_path, cmd_args.method, cmd_args.g_type, f'temp-{2 * cmd_args.num_leaves}-losses.ckpt')
        loss_data = {'loss_tops': loss_tops_list, 'loss_wts': loss_wts_list}
        torch.save(loss_data, path)

    print("Training Complete") 
    print("Evaluating on Test Graphs...")
    gen_graphs = []
    
    with torch.no_grad():
        model.eval()
        num_node_dist = get_node_dist(train_graphs)
        np.random.seed(cmd_args.seed) 
        num_nodes_list = [np.random.choice(len(num_node_dist), p=num_node_dist) for _ in range(20)]
        print(num_nodes_list)
        
        for idx in tqdm(range(20)):
            num_nodes = num_nodes_list[idx]
            _, pred_edges, _, _, pred_edge_feats = model(node_end = num_nodes, display=cmd_args.display)
            
            if cmd_args.method == 'BiGG-GCN':
                fix_edges = [(min(e1, e2), max(e1, e2)) for e1, e2 in pred_edges]
                pred_edge_tensor = torch.tensor(fix_edges).to(cmd_args.device)
                pred_weighted_tensor = model.gcn_mod.sample(num_nodes, pred_edge_tensor).cpu().numpy()
                weighted_edges = [(int(e1), int(e2), round(w.item(), 4)) for e1, e2, w in pred_weighted_tensor]
            
            else:
                weighted_edges = [(min(e[0], e[1]), max(e[0], e[1]), round(w.item(), 4)) for e, w in zip(pred_edges, pred_edge_feats)]
                
            pred_g = nx.Graph()
            pred_g.add_weighted_edges_from(weighted_edges)
            gen_graphs.append(pred_g)
            print(correct_tree_topology_check_two([pred_g]))
            
    print("Getting Graph Stats...")
    get_graph_stats(gen_graphs, test_graphs, 'scale_test', cmd_args.g_type)




