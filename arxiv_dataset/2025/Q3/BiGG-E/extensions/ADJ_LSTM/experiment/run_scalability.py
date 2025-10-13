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
from extensions.ADJ_LSTM.model.LSTM_models import *
from extensions.common.configs import cmd_args, set_device
from extensions.ADJ_LSTM.util.train_util import *
from extensions.evaluate.mmd import *
from extensions.evaluate.mmd_stats import *
from bigg.torch_ops.tensor_ops import *
from bigg.common.consts import t_float
from extensions.synthetic_gen.data_util import *
from extensions.synthetic_gen.data_creator import tree_generator, graph_generator


if __name__ == '__main__': 
    random.seed(cmd_args.seed)
    torch.manual_seed(cmd_args.seed)
    np.random.seed(cmd_args.seed)
    set_device(cmd_args.gpu)
    cmd_args.max_num_nodes = 2 * cmd_args.num_leaves - 1
    
    print("Getting training times")
    cmd_args.num_graphs = 1
    num_leaves = cmd_args.num_leaves
    print("Number of Leaves: ", num_leaves)
    print("Method: ", "ADJ-LSTM")
    
    num_nodes = 2 * int(num_leaves) - 1
    m = num_nodes - 1
    cmd_args.num_leaves = num_leaves
    g = graph_generator(cmd_args)
    g = get_graph_data(g[0], 'BFS', global_source=0)
    g = g[0]
    cmd_args.device = "cuda"
    batched_graphs = batch_preparation([g], True).to("cuda") 
    times = []
    memories = []
    
    for i in range(1):
        model = AdjacencyLSTM(cmd_args).to("cuda")
        optimizer = torch.optim.AdamW(model.parameters(), lr = cmd_args.learning_rate, weight_decay = 1e-4)
        
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        init_adj = datetime.now()
        loss_r, loss_w = model(batched_graphs)
        loss = loss_r + loss_w
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        torch.cuda.synchronize()
        cur_adj = datetime.now() - init_adj
        
        mem_used = torch.cuda.max_memory_allocated() / 1024**2 
        print("mem: ", mem_used)
                
        if i >= 0:
            times.append(cur_adj.total_seconds())
            memories.append(mem_used)
        
        del model
        del optimizer
        torch.cuda.empty_cache()
        gc.collect()
    
    print("Times: ", times)
    print("Average Time in 7 runs: ", np.mean(times))
    print("Average Peak Memory in 7 runs: ", np.mean(memories))
    sys.exit()