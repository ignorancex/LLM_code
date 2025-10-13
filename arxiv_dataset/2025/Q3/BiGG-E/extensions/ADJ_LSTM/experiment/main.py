from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import cmd

import networkx as nx
import pickle as cp
import numpy as np
import torch
import random
from torch import nn
from torch.nn.parameter import Parameter
from tqdm import tqdm
import pandas as pd
import os
from scipy.stats.distributions import chi2
import sys
from extensions.ADJ_LSTM.model.LSTM_models import *
from extensions.evaluate.graph_stats import *
from extensions.common.configs import cmd_args, set_device
from extensions.ADJ_LSTM.util.train_util import *
from extensions.evaluate.mmd import *
from extensions.evaluate.mmd_stats import *
from bigg.torch_ops.tensor_ops import *
from bigg.common.consts import t_float
from extensions.BiGG_E.model_extensions.util_extension.train_util import *
from extensions.synthetic_gen.data_util import *
from extensions.synthetic_gen.data_creator import *
# from bigg.experiments.train_utils import get_node_dist



def extract_weights_with_stats(graphs):
    weight_lists = []
    mins, q1s, medians, q3s, maxs = [], [], [], [], []
    means, stds = [], []
    
    for G in graphs:
        weights = [data['weight'] for _, _, data in G.edges(data=True) if 'weight' in data]
        if not weights:
            # Handle graphs with no weighted edges
            weights = [np.nan]

        weights_np = np.array(weights)
        weight_lists.append(weights)

        mins.append(np.min(weights_np))
        q1s.append(np.percentile(weights_np, 25))
        medians.append(np.median(weights_np))
        q3s.append(np.percentile(weights_np, 75))
        maxs.append(np.max(weights_np))
        means.append(np.mean(weights_np))
        stds.append(np.std(weights_np, ddof=1))

    return {
        "weights": weight_lists,
        "min": mins,
        "q1": q1s,
        "median": medians,
        "q3": q3s,
        "max": maxs,
        "mean": means,
        "std": stds
    }



if __name__ == '__main__':
    random.seed(cmd_args.seed)
    torch.manual_seed(cmd_args.seed)
    np.random.seed(cmd_args.seed)
    
    set_device(cmd_args.gpu)
    
    ## Opening and Preparing Training Graphs
#     scale = True
#     if scale:
#         print("Scale Run")
#         cmd_args.num_graphs = 100
#         cmd_args.g_type = "tree"
#         cmd_args.graph_type = "tree"
#         graphs = create_training_graphs(cmd_args)
#         
#         cmd_args.g_type = "tree"
#         ordered_train_graphs, test_graphs = process_graphs(graphs, 'DFS')
#         cmd_args.batch_size = 10
#         cmd_args.accum_grad = 2
#         cmd_args.num_epochs = 10
#         cmd_args.epoch_save = 10
    
    else:
        path = os.path.join(cmd_args.data_dir, '%s-graphs.pkl' % 'train')
        with open(path, 'rb') as f:
            ordered_train_graphs = cp.load(f) ## List of nx train graphs
    
        path = os.path.join(cmd_args.data_dir, '%s-graphs.pkl' % 'val')
        with open(path, 'rb') as f:
            val_graphs = cp.load(f) ## List of nx val graphs
    
    num_node_dist = get_node_dist(ordered_train_graphs)
    if sum(num_node_dist > 0) == 1:
        cmd_args.constant_nodes = True
        
    cmd_args.max_num_nodes = len(num_node_dist) - 1
    
    print("Constant Nodes Check: ", cmd_args.constant_nodes)
    print("Max num nodes: ", cmd_args.max_num_nodes)
    
    print("Number of training graphs: ", len(ordered_train_graphs))
    
    if cmd_args.phase == 'train':
        batched_train_graphs = batch_preparation(ordered_train_graphs, cmd_args.weighted).to(cmd_args.device) ## Converted to (padded) tensor
    
    ## Creating and Loading Model (if applciable)
    
    if cmd_args.baseline:
        from extensions.ADJ_LSTM.util.baseline_train import *
        print("Baseline Run")
        
        num_gen = {'er': 20, 'db': 9, 'tree': 200, 'lobster': 200, 'joint': 20, 'joint_2': 20}
        cmd_args.num_gen = num_gen[cmd_args.graph_type]
        
        num_nodes = []
        num_edges = []
        for g in ordered_train_graphs:
            num_nodes.append(len(g))
            num_edges.append(len(g.edges()))
        print("Num Nodes: ", np.mean(num_nodes), (min(num_nodes), max(num_nodes)))
        print("Num Edges: ", np.mean(num_edges), (min(num_edges), max(num_edges)))
        
        print("Using Generator: ", cmd_args.generator)
        out_er_graphs = Graph_generator_baseline(batched_train_graphs, cmd_args.weighted, cmd_args.generator, cmd_args.num_gen)
        out_graphs = []
        
        for graph in out_er_graphs:
            if len(graph.edges()) > 0:
                out_graphs.append(graph)
        print('getting graph statistics...')
        
        wts = []
        for graph in out_er_graphs:
            cur = [w['weight'] for _, _, w in graph.edges(data=True)]
            wts += [np.var(cur, ddof = 1)]
        print("HERE HERE")
        print(wts)
        
        path = os.path.join(cmd_args.data_dir, '%s-graphs.pkl' % 'test')
        with open(path, 'rb') as f:
            test_graphs = cp.load(f) 
        
        get_graph_stats(out_graphs, test_graphs, cmd_args.graph_type)
        print('graph statistics complete!')
        sys.exit()
    
    model = AdjacencyLSTM(cmd_args).to(cmd_args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr = cmd_args.learning_rate, weight_decay = 1e-4)
    
    if cmd_args.epoch_load is not None:
        cmd_args.model_dump = f'../../model_saves/{cmd_args.graph_type}/epoch-{cmd_args.epoch_load}.ckpt'
    
    print(cmd_args.model_dump)
    
    if cmd_args.model_dump is not None and os.path.isfile(cmd_args.model_dump):
        print('loading from', cmd_args.model_dump)
        checkpoint = torch.load(cmd_args.model_dump)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
    
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    
    ## Sampling Section
    if cmd_args.phase == "validate":
        with torch.no_grad():
            model.eval()
            out_graphs = []
            for i in tqdm(range(cmd_args.num_gen)):
                num_nodes = np.argmax(np.random.multinomial(1, num_node_dist))
                v = model.predict(num_nodes, tol = cmd_args.tol)
                g = graph_from_adj(v)
                if len(g.edges()) > 0:
                    out_graphs.append(g)
                
                else:
                    print("EMPTY GRAPH")
        
        print('getting graph statistics...')
        path = os.path.join(cmd_args.data_dir, '%s-graphs.pkl' % 'val')
        
        with open(path, 'rb') as f:
            val_graphs = cp.load(f)
        
        get_graph_stats(out_graphs, val_graphs, cmd_args.graph_type)
        print('graph statistics complete!')
        sys.exit()
    
    elif cmd_args.phase == "test":
        with torch.no_grad():
            model.eval()
            out_graphs = []
            for i in tqdm(range(cmd_args.num_gen)):
                num_nodes = np.argmax(np.random.multinomial(1, num_node_dist))
                v = model.predict(num_nodes, tol = cmd_args.tol)
                g = graph_from_adj(v)
                if i <= 4:
                    print(g.edges(data=True))
                out_graphs.append(g)
        
        if cmd_args.graph_type == "tree":
            print("Save Tree Weights")
            gen_stats = extract_weights_with_stats(out_graphs)
            
            with open(cmd_args.model_dump + '.gen_stats', 'wb') as f:
                cp.dump(gen_stats, f, cp.HIGHEST_PROTOCOL)
        
        print('getting graph statistics...')
        path = os.path.join(cmd_args.data_dir, '%s-graphs.pkl' % 'test')
        with open(path, 'rb') as f:
            test_graphs = cp.load(f) 
        
        get_graph_stats(out_graphs, test_graphs, cmd_args.graph_type)
        print('graph statistics complete!')
        sys.exit()
     
    ## Training through epoch number
    if cmd_args.epoch_load is None:
        cmd_args.epoch_load = 0
    
    print("Begin Training...")
    
    N = batched_train_graphs.shape[0]
    B = cmd_args.batch_size
    indices = list(range(N))
    num_iter = int(N / B)
    model.train()
    
    times = []
    loss_times = []
    epoch_list = []
    prev = datetime.now()
    
    lr_scheduler = {'lobster': 50, 'tree': 50, 'er': 30} # , 'db': 1000, 'er': 250}
    epoch_lr_decrease = lr_scheduler[cmd_args.graph_type]
    
    for epoch in range(cmd_args.epoch_load, cmd_args.num_epochs):
        if epoch >= epoch_lr_decrease and cmd_args.learning_rate != 1e-5:
            cmd_args.learning_rate = 1e-5
            for param_group in optimizer.param_groups:
                param_group['lr'] = 1e-5
        
        tot_loss = 0.0 ### for grad_accum
        pbar = tqdm(range(num_iter))
        np.random.shuffle(indices)
        model.epoch_num += 1
        
        for idx in pbar:
          if idx >= cmd_args.accum_grad * int(num_iter / cmd_args.accum_grad):
              print("Skipping iteration -- not enough sub-batches remaining for grad accumulation.")
              continue
          
          start = B * idx
          stop = B * (idx + 1)
          batch_indices = indices[start:stop]
          
          batched_graphs = batched_train_graphs[batch_indices, :, :]
          
          num_nodes = compute_num_nodes(batched_graphs)
          
          loss_r, loss_w = model(batched_graphs)
          loss = loss_r + loss_w
          loss.backward()
          loss = loss.item()
          tot_loss = tot_loss + loss / cmd_args.accum_grad
          
          if (idx + 1) % cmd_args.accum_grad == 0:                        
              cur = datetime.now() - prev
              times.append(cur.total_seconds())
              loss_times.append(tot_loss)
              epoch_list.append(epoch)
              
              tot_loss = 0.0
              
              if cmd_args.grad_clip > 0:
                  torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cmd_args.grad_clip)
              optimizer.step()
              optimizer.zero_grad()
          
          pbar.set_description('epoch %.2f, loss: %.4f' % (epoch + (idx + 1) / num_iter, loss))
        
        cur = epoch + 1
        print('Epoch Complete')
        if (epoch + 1) % cmd_args.epoch_save == 0 or epoch + 1 == cmd_args.num_epochs:
            print('Saving Epoch')
            checkpoint = {'epoch': epoch, 'model': model.state_dict(), 'optimizer': optimizer.state_dict()}
            torch.save(checkpoint, os.path.join(cmd_args.save_dir, 'epoch-%d.ckpt' % (epoch + 1)))
    
    if scale:
        model.eval()
        import time
        with torch.no_grad():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            start = time.time()
            v = model.predict(2 * cmd_args.num_leaves - 1, tol = cmd_args.tol)
            torch.cuda.synchronize()
            elapsed = time.time() - start
            gc.collect()
            torch.cuda.empty_cache()
        print(f"Time to sample 1 graph: {elapsed:.4f} seconds")
    
    elapsed = datetime.now() - prev
    print("Time elapsed during training: ", elapsed)
    
    print("Model Training Complete")
    time_data = {'times': times, 'loss_times': loss_times, 'epoch_list': epoch_list}
    
    with open('%s-time-data.pkl' % cmd_args.graph_type, 'wb') as f:
        cp.dump(time_data, f, protocol=cp.HIGHEST_PROTOCOL)
    












