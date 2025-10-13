# -*- coding: utf-8 -*-
"""
Created on Mon May 12 13:19:57 2025

@author: Molly
"""

import torch
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.datasets import Reddit
from torch_geometric.datasets import Planetoid
from cogdl.data import Graph
import numpy as np
import scipy.sparse as sp
import json



class GraphData():
    def __init__(self,dataset_name):
        self.dataset_name=dataset_name
    
        
        if dataset_name=='ogbn-arxiv':
            self.hidden_size=256
            self.weight_decay=0.00001
            self.dropout=0
            self.epochs=150
            
        if dataset_name=='flickr':
            self.hidden_size=256
            self.weight_decay=0
            self.dropout=0.2
            self.epochs=200
            
        if dataset_name=='Cora' or dataset_name=='CiteSeer' or dataset_name=='PubMed':
            self.hidden_size=32
            self.weight_decay=0.0005
            self.dropout=0.5
            self.epochs=100

        if dataset_name=='Reddit':
            self.hidden_size=128
            self.weight_decay=0
            self.dropout=0
            self.epochs=30

        if dataset_name=='ppi':
            self.hidden_size=256
            self.weight_decay=0
            self.dropout=0
            self.epochs=200

        if dataset_name=='ppi-large':
            self.hidden_size=256
            self.weight_decay=0
            self.dropout=0
            self.epochs=200
            
        if dataset_name=='yelp':
            self.hidden_size=256
            self.weight_decay=0
            self.dropout=0.1
            self.epochs=200


        
    def data_preprocessing(self):
        if self.dataset_name =='flickr': #load in from GraphSAINT files
            adj_sparse=sp.load_npz(self.dataset_name+'/adj_full.npz')
            
            #edge index for adj mat
            adj_coo=adj_sparse.tocoo()
            r=adj_coo.row
            c=adj_coo.col
            tensor_edge_index=torch.tensor([r,c],dtype=torch.long) 
            
            #features
            feat=np.load(self.dataset_name+'/feats.npy')
            feature_mat=torch.tensor(feat,dtype=torch.float32) 
            
            #classes
            with open(self.dataset_name+'/class_map.json','r') as file:
                class_data=json.load(file)
            class_list=list(class_data.values())
            class_tensor=torch.tensor(class_list)
            
            #making Cogdl graph object
            data=Graph(edge_index=tensor_edge_index,x=feature_mat,y=class_tensor) #has no self loops at this point
            
            #setting number of in channels and out channels
            num_in_channels=feature_mat.shape[1]
            num_out_channels=max(class_list)+1
            
            #train/test/val split
            with open(self.dataset_name+'/role.json','r') as file:
                data_split_dict=json.load(file)
            train_ind_list=data_split_dict['tr']
            test_ind_list=data_split_dict['te']
            val_ind_list=data_split_dict['va']
            
            #train,test,val masks
            train_mask = torch.tensor([False] * len(class_tensor))
            train_mask[train_ind_list]= True
            data.train_mask = train_mask
            
            test_mask = torch.tensor([False] * len(class_tensor))
            test_mask[test_ind_list]= True
            data.test_mask = test_mask
            
            val_mask = torch.tensor([False] * len(class_tensor))
            val_mask[val_ind_list]= True
            data.val_mask = val_mask
            
            
    ############################################################################
        
        else:
            if self.dataset_name=='ogbn-arxiv':
                dataset = PygNodePropPredDataset(name=self.dataset_name)
            elif self.dataset_name=='Reddit':
                dataset=Reddit(root="data")
            elif self.dataset_name=='Cora' or self.dataset_name=='CiteSeer' or self.dataset_name=='PubMed': #for Planetoid
                dataset=Planetoid(root="data", name=self.dataset_name, split="public")      
            
            num_in_channels=dataset.num_features
            num_out_channels=dataset.num_classes
            
            #make CogDL graph
            g=dataset[0]
            data=Graph(edge_index=g.edge_index,x=g.x,y=g.y) #has no self loops at this point
        
            if self.dataset_name=='ogbn-arxiv':
                #train, test, val masks
                split_idx = dataset.get_idx_split()
                y=g.y
            
                training_mask = torch.tensor([False] * len(y))
                training_mask[split_idx['train']] = True
                data.train_mask=training_mask
            
                test_mask = torch.tensor([False] * len(y))
                test_mask[split_idx['test']]= True
                data.test_mask = test_mask
            
                val_mask = torch.tensor([False] * len(y))
                val_mask[split_idx['valid']] = True
                data.val_mask = val_mask
                
            else: #for Planetoid and Reddit
                #set masks for Planetoid datasets
                data.train_mask=g.train_mask
                data.val_mask=g.val_mask
                data.test_mask=g.test_mask
                
                print(f"num train nodes: {data.train_mask.sum()}")
                print(f"num test nodes: {data.test_mask.sum()}")
                print(f"num val nodes: {data.val_mask.sum()}")
        
        
        ############check if data is symmetric###################
        edge_index=data.edge_index
        edge_index=torch.stack(edge_index,dim=0)
        reversed_edge_index = torch.stack([edge_index[1], edge_index[0]], dim=0)
        symmetric=set(map(tuple, edge_index.t().tolist())) == set(map(tuple, reversed_edge_index.t().tolist()))
        print(f"{self.dataset_name} symmetric:{symmetric}")
        
        #adding self_loops and symmetric normalization
        data.add_remaining_self_loops()
        
        #symmetric normalization of the adjacency matrix
        degrees=data.degrees() #diag of D matrix
        degrees_inv_sqrt=degrees.pow(-0.5)
        degrees_inv_sqrt[degrees_inv_sqrt==float("inf")]=0
        edge_weights=data.edge_weight
        row,col=data.edge_index
        data.edge_weight=degrees_inv_sqrt[col]*edge_weights*degrees_inv_sqrt[row]
        
        #dataset info
        print(data)
        
        return data, num_in_channels, num_out_channels 
    
        
    
    
    
    
    
