#####################################
#          FEED FORWARD NN          #
#####################################

import os
import json
from importlib import util

import torch
import torch.nn as nn

import tensorkrowch as tk


cwd = os.getcwd()

def import_file(full_name, path):
    """Returns a python module given its path"""
    spec = util.spec_from_file_location(full_name, path)
    mod = util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def permuted_dims(dim):
    """
    Returns permutation of matrix dimensions, given the matrix is in tensor
    form  with ``dim`` input legs and ``dim`` output legs.
    """
    dims = []
    for i in range(dim):
        dims += [dim + i, i]
    return dims


model_name = 'fffc_medium'

aux_mod = import_file('model', os.path.join(cwd, 'models', f'{model_name}.py'))
model_class = aux_mod.Model

# Check tuned config of balanced model
config_dir = os.path.join(cwd, 'results', '0_train_nns', model_class.name, '0.5')
with open(os.path.join(config_dir, 'tuned_config.json'), 'r') as f:
    config = json.load(f)
        
models_dir = os.path.join(cwd, 'results', '2_compression')

# Load state_dict
state_dict_dir = list(filter(lambda f: f.startswith(model_name) and f.endswith('.pt'),
                             os.listdir(models_dir)))[0]
state_dict = torch.load(os.path.join(models_dir, state_dict_dir),
                        weights_only=False)

# Initialize model with balanced config
model = model_class(config)
model.load_state_dict(state_dict)


class TN_Linear1(tk.models.MPO):
    
    def __init__(self, model, bond_dim):
        
        # Get weight matrix from model and reshape it
        weight = model.layers[0].weight.detach()
        weight = weight.reshape(2, 5, 5, 2, 1,
                                2, 5, 5, 5, 2).permute(permuted_dims(5))
        self.weight = weight
        
        mpo_tensors = tk.decompositions.mat_to_mpo(weight,
                                                   rank=bond_dim,
                                                   cum_percentage=0.99,
                                                   renormalize=True)
        super().__init__(tensors=mpo_tensors)
        
        # Save bias as parameter of tn layer
        self.bias = nn.Parameter(model.layers[0].bias.detach())
    
    def set_data_nodes(self):
        self.data_node = tk.Node(shape=(1, 2, 5, 5, 5, 2),
                                 axes_names=(('batch', *(['feature'] * 5))),
                                 name='data',
                                 network=self,
                                 data=True)
        for data_edge, mpo_node in zip(self.data_node.edges[1:],
                                       self.mats_env):
            data_edge ^ mpo_node['input']
    
    def add_data(self, data):
        data = data.view(-1, 2, 5, 5, 5, 2)
        self.data_node.tensor = data
    
    def contract(self):
        nodes = self.mats_env
        nodes[0] = self.left_node @ nodes[0]
        nodes[-1] = nodes[-1] @ self.right_node
        
        result = self.data_node
        for node in nodes:
            result @= node
        
        return result
        
    def forward(self, x, *args, **kwargs):
        result = super().forward(x, *args, **kwargs)
        result = result.reshape(-1, 100)
        result += self.bias
        return result


class TN_Linear2(tk.models.MPO):
    
    def __init__(self, model, bond_dim):
        
        # Get weight matrix from model and reshape it
        weight = model.layers[3].weight.detach()
        weight = weight.reshape(2, 5, 2, 1,
                                2, 5, 5, 2).permute(permuted_dims(4))
        self.weight = weight
        
        mpo_tensors = tk.decompositions.mat_to_mpo(weight,
                                                   rank=bond_dim,
                                                   cum_percentage=0.99,
                                                   renormalize=True)
        super().__init__(tensors=mpo_tensors)
        
        # Save bias as parameter of tn layer
        self.bias = nn.Parameter(model.layers[3].bias.detach())
    
    def set_data_nodes(self):
        self.data_node = tk.Node(shape=(1, 2, 5, 5, 2),
                                 axes_names=(('batch', *(['feature'] * 4))),
                                 name='data',
                                 network=self,
                                 data=True)
        for data_edge, mpo_node in zip(self.data_node.edges[1:],
                                       self.mats_env):
            data_edge ^ mpo_node['input']
    
    def add_data(self, data):
        data = data.view(-1, 2, 5, 5, 2)
        self.data_node.tensor = data
    
    def contract(self):
        nodes = self.mats_env
        nodes[0] = self.left_node @ nodes[0]
        nodes[-1] = nodes[-1] @ self.right_node
        
        result = self.data_node
        for node in nodes:
            result @= node
        
        return result
        
    def forward(self, x, *args, **kwargs):
        result = super().forward(x, *args, **kwargs)
        result = result.reshape(-1, 20)
        result += self.bias
        return result


class TN_Linear3(tk.models.MPO):
    
    def __init__(self, model, bond_dim):
        
        # Get weight matrix from model and reshape it
        weight = model.layers[6].weight.detach()
        weight = weight.reshape(1, 2, 1,
                                2, 5, 2).permute(permuted_dims(3))
        self.weight = weight
        
        mpo_tensors = tk.decompositions.mat_to_mpo(weight,
                                                   rank=bond_dim,
                                                   cum_percentage=0.99,
                                                   renormalize=True)
        super().__init__(tensors=mpo_tensors)
        
        # Save bias as parameter of tn layer
        self.bias = nn.Parameter(model.layers[6].bias.detach())
    
    def set_data_nodes(self):
        self.data_node = tk.Node(shape=(1, 2, 5, 2),
                                 axes_names=(('batch', *(['feature'] * 3))),
                                 name='data',
                                 network=self,
                                 data=True)
        for data_edge, mpo_node in zip(self.data_node.edges[1:],
                                       self.mats_env):
            data_edge ^ mpo_node['input']
    
    def add_data(self, data):
        data = data.view(-1, 2, 5, 2)
        self.data_node.tensor = data
    
    def contract(self):
        nodes = self.mats_env
        nodes[0] = self.left_node @ nodes[0]
        nodes[-1] = nodes[-1] @ self.right_node
        
        result = self.data_node
        for node in nodes:
            result @= node
        
        return result
        
    def forward(self, x, *args, **kwargs):
        result = super().forward(x, *args, **kwargs)
        result = result.reshape(-1, 2)
        result += self.bias
        return result


class Model(nn.Module):
    
    name = model_name + '_tn'
    
    def __init__(self, bond_dim):
        super().__init__()
        
        p_do = config['p_do']
        
        self.layers = nn.Sequential(
            TN_Linear1(model, bond_dim),
            nn.Dropout(p_do),
            nn.ReLU(),
            TN_Linear2(model, bond_dim),
            nn.Dropout(p_do),
            nn.ReLU(),
            TN_Linear3(model, bond_dim))
        
        self.layers[0].trace(torch.zeros(1, 500))
        self.layers[3].trace(torch.zeros(1, 100))
        self.layers[6].trace(torch.zeros(1, 20))
        
    def forward(self, x):
        return self.layers(x)
