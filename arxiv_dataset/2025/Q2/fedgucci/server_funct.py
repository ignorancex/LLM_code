from typing import Dict, List
import numpy as np
import torch
import torch.nn.functional as F
import os
import random
from torch.backends import cudnn
from random import sample
import math
import torch.optim as optim
import torch.nn as nn
from nodes import Node
from utils import load_data
import copy
from torch.optim.lr_scheduler import CosineAnnealingLR
from utils import init_model, trainable_params

##############################################################################
# General server function
##############################################################################

def receive_client_models(args, client_nodes, select_list, size_weights):
    client_params = []
    client_caches = []
    for idx in select_list:
        if 'fedawo' in args.server_method:
            client_params.append(client_nodes[idx].model.get_param(clone = True))

        elif args.client_method == 'mask_intrinsic':
            client_params.append(copy.deepcopy(client_nodes[idx].model.state_dict()))
            client_caches.append(copy.deepcopy(client_nodes[idx].model.m[0].state_dict()))

        else:
            client_params.append(copy.deepcopy(client_nodes[idx].model.state_dict()))
    
    agg_weights = [size_weights[idx] for idx in select_list]
    agg_weights = [w/sum(agg_weights) for w in agg_weights]

    if args.client_method == 'mask_intrinsic':
        return agg_weights, [client_params, client_caches]
    else:
        return agg_weights, client_params


##############################################################################
# Our method's function
##############################################################################

def generate_activation_mask(args, central_node):
    mask = []
    reverse_mask = []
    for p_idx, p in enumerate(central_node.model.parameters()):
        p_grad = p.reshape(-1)
        p_mask = torch.from_numpy(np.random.choice([0, 1], size=p_grad.size()[0], p=[args.mask_ratio, 1-args.mask_ratio])).cuda()
        p_mask_reversed = 1 - p_mask
        mask.append(p_mask)
        reverse_mask.append(p_mask_reversed)
    central_node.mask = mask
    central_node.reverse_mask = reverse_mask
    return central_node, mask

def generate_layer_wise_activation_mask(args, central_node, mask_layer_name):
    '''
    mask_layer_name: last, first, middle
    '''
    mask = []
    reverse_mask = []
    for p_idx, (name, p) in enumerate(central_node.model.named_parameters()):
        p_grad = p.reshape(-1)
        if args.local_model == 'ResNet20':
            name_dict = {'last':'layer_5' , 'first':'layer_1', 'middle':'layer_3'}
        elif args.local_model == 'CNN':
            name_dict = {'last':'layer_5' , 'first':'layer_1', 'middle':'layer_3', 'last2':'layer_4'}
        elif args.local_model == 'MLP_h2_w200':
            name_dict = {'last':'layer_2' , 'first':'layer_0', 'middle':'layer_1'}


        layer_name = name_dict[mask_layer_name]
        if layer_name in name:
            p_mask = torch.from_numpy(np.random.choice([0, 1], size=p_grad.size()[0], p=[args.mask_ratio, 1-args.mask_ratio])).cuda()
        else:
            p_mask = torch.from_numpy(np.array([1 for _ in range(p_grad.size()[0])])).cuda()

        p_mask_reversed = 1 - p_mask
        mask.append(p_mask)
        reverse_mask.append(p_mask_reversed)
    central_node.mask = mask
    central_node.reverse_mask = reverse_mask
    return central_node, mask



def generate_fastslow_mask(args, central_node):
    mask = []
    reverse_mask = []
    for p_idx, p in enumerate(central_node.model.parameters()):
        p_grad = p.reshape(-1)
        p_mask = torch.from_numpy(np.random.choice([0.2, 1], size=p_grad.size()[0], p=[args.mask_ratio, 1-args.mask_ratio])).cuda()
        p_mask_reversed = 1 - p_mask
        mask.append(p_mask)
        reverse_mask.append(p_mask_reversed)
    central_node.mask = mask
    central_node.reverse_mask = reverse_mask
    return central_node, mask


def prune_model_with_mask(mask, central_node):
    for p_idx, p in enumerate(central_node.model.parameters()):
        p_size = p.data.size()
        p_data = p.data.reshape(-1)
        
        p_mask = mask[p_idx]

        p_data = p_data * p_mask
        p.data = p_data.view(p_size)
    
    return central_node


def generate_direction_mask(args, central_node):
    mask = []
    for p_idx, p in enumerate(central_node.model.parameters()):
        p_grad = p.reshape(-1)
        p_mask = torch.from_numpy(np.random.choice([-1, 1], size=p_grad.size()[0], p=[args.negative_ratio, 1-args.negative_ratio])).cuda()
        mask.append(p_mask)
    central_node.direc_mask = mask
    return central_node, mask

def generate_gaussian_mask(args, central_node):
    mask = []
    for p_idx, p in enumerate(central_node.model.parameters()):
        p_grad = p.reshape(-1)
        p_mask = torch.normal(mean=torch.tensor([1.0 for _ in range(p_grad.size()[0])]), std=args.gaussian_std).cuda()
        mask.append(p_mask)
    central_node.gauss_mask = mask
    return central_node, mask

##############################################################################
# Baselines function (FedAvg, Fedprox, etc.)
##############################################################################

# generate random global model set for each round
def sample_simplex(n):
    """Sample a point from an n-dimensional simplex."""
    # Sample n random points between 0 and 1
    points = np.random.rand(n-1)
    
    # Add 0 and 1 to the points and then sort them
    points = np.sort(np.concatenate(([0], points, [1])))
    
    # Take the difference between consecutive points to get the coordinates
    # of a point in the simplex
    weights =  np.diff(points).tolist()
    return weights

def generate_global_model_set(args, central_node, client_nodes, select_list, size_weights):
    central_node.curr_round_global_model_set = []
    _, client_params = receive_client_models(args, client_nodes, select_list, size_weights)
    model = copy.deepcopy(central_node.model)

    for num in range(args.group_model_num):
        weights = sample_simplex(args.node_num)
        state_dict = fedavg(client_params, weights)
        model.load_state_dict(state_dict)
        central_node.curr_round_global_model_set.append(copy.deepcopy(model))
    
    return central_node

    
   


def Server_update(args, central_node, client_nodes, select_list, size_weights):
    '''
    server update functions for baselines
    '''

    # receive the local models from clients
    agg_weights, client_params = receive_client_models(args, client_nodes, select_list, size_weights)

    # update the global model
    if args.server_method == 'fedavg':
        if args.client_method == 'mask_intrinsic':
            client_params, client_caches = client_params[0], client_params[1]
            avg_global_param = fedavg(client_params, agg_weights)
            avg_global_cache = fedavg(client_caches, agg_weights)
            
            # set and load V
            central_node.model.load_state_dict(avg_global_param)
            # # set and load cache
            central_node.model.m[0].load_state_dict(avg_global_cache)


            # For 每次round都是一个新的起点
            # set m according to new V and previous initial
            central_node.model.set_m()
            # set new initial
            model_copy = central_node.model
            central_node.model.set_initial(model_copy)

        else:
            avg_global_param = fedavg(client_params, agg_weights)
            central_node.model.load_state_dict(avg_global_param)

    elif args.server_method == 'feddyn':
        central_node = feddyn(args, central_node, agg_weights, client_nodes, select_list)
    
    elif args.server_method == 'scaffold':
        central_node = scaffold(args, central_node, client_nodes)
    
    elif args.server_method == 'none': # don't aggregat model
        pass

    else:
        raise ValueError('Undefined server method...')

    return central_node

def fedavg(parameters, list_nums_local_data):
    fedavg_global_params = copy.deepcopy(parameters[0])
    for name_param in parameters[0]:
        list_values_param = []
        for dict_local_params, num_local_data in zip(parameters, list_nums_local_data):
            list_values_param.append(dict_local_params[name_param] * num_local_data)
        value_global_param = sum(list_values_param) / sum(list_nums_local_data)
        fedavg_global_params[name_param] = value_global_param
    return fedavg_global_params

def feddyn(args, central_node, agg_weights, client_nodes, select_list):
    '''
    server function for feddyn
    '''

    # update server's state
    uploaded_models = []
    for i in select_list:
        uploaded_models.append(copy.deepcopy(client_nodes[i].model))

    model_delta = copy.deepcopy(uploaded_models[0])
    for param in model_delta.parameters():
        param.data = torch.zeros_like(param.data)

    for idx, client_model in enumerate(uploaded_models):
        for server_param, client_param, delta_param in zip(central_node.model.parameters(), client_model.parameters(), model_delta.parameters()):
            delta_param.data += (client_param - server_param) * agg_weights[idx]

    for state_param, delta_param in zip(central_node.server_state.parameters(), model_delta.parameters()):
        state_param.data -= args.mu * delta_param

    # aggregation
    central_node.model = copy.deepcopy(uploaded_models[0])
    for param in central_node.model.parameters():
        param.data = torch.zeros_like(param.data)
        
    for idx, client_model in enumerate(uploaded_models):
        for server_param, client_param in zip(central_node.model.parameters(), client_model.parameters()):
            server_param.data += client_param.data.clone() * agg_weights[idx]

    for server_param, state_param in zip(central_node.model.parameters(), central_node.server_state.parameters()):
        server_param.data -= (1/args.mu) * state_param

    return central_node

def scaffold(args, central_node: Node, client_nodes: Dict[int, Node]):
    y_delta_cache = []
    c_delta_cache = []
    for client_node in client_nodes.values():
        y_delta_cache.append(client_node.y_delta)
        c_delta_cache.append(client_node.c_delta)
    
    for param, y_delta in zip(
        trainable_params(central_node.model), zip(*y_delta_cache)
    ):
        param.data.add_(
            args.server_lr * torch.stack(y_delta, dim=-1).mean(dim=-1)
        )

    # update global control
    for c_global, c_delta in zip(central_node.c_global, zip(*c_delta_cache)):
        c_global.data.add_(
            torch.stack(c_delta, dim=-1).sum(dim=-1) / len(client_nodes)
        )
    
    return central_node