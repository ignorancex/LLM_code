import torch
from torch.utils.data import Dataset
from scipy import sparse
import numpy as np
import dgl 
from dgl.data.utils import load_graphs
import h5py
dgl.use_libxsmm(False)

class GraphData_Loader(Dataset):
    def __init__(self, data_dict, label_dict, phase, transform=None, magnifications=['5x', '10x', '20x'], super_patch=False, pooling='mean', graph_type = None):
        self.data_dict = data_dict # dictionary with keys: feature_path, label, slide_id
        self.phase = phase # 'train', 'val', 'test'
        self.transform=transform
        self.magnifications = magnifications
        self.super_patch = super_patch # for TCGA project it is True by default, unless it is specific to a project which False
        self.pooling = pooling # mean, median, max pooling or None as no poooling for patches 
        self.label_dict = label_dict # dictionary with keys: labels assignment 
        self.graph_type = graph_type
    def __len__(self):
        return len(self.data_dict['feature_path'][self.phase])
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        #print(idx)    
        path_to_dict = self.data_dict['feature_path'][self.phase][idx]
        if path_to_dict.split('.')[-1] == 'bin':
            data = load_graphs(path_to_dict)[0][0]
            data.ndata['x'] = data.ndata['x'].to(torch.float32)
            if 'c' in data.ndata.keys():
                data.ndata['c'] = data.ndata['c'].to(torch.float32).squeeze()
            else:
                print("missing coords for:", path_to_dict)
        elif path_to_dict.split('.')[-1] == 'h5':
            from Benchmarking.utils.grasp_utils import build_graph
            f = h5py.File(path_to_dict, 'r')
            feat = torch.tensor(f['features'][self.magnifications[0]]).to(torch.float32)
            data = {}
            coor = {}
            for key in self.magnifications:
                data[key] = torch.tensor(f['features'][key][:]).to(torch.float32)
                coor[key] = torch.tensor(f['coords'][key][:]).to(torch.float32)
            f.close()
            f = None
            feats = {'features': data, 'coords': coor} 
            data = build_graph(feats)
        
        #print(data.ndata['c'])
        if len(data.ndata['x'])%3 != 0:
            raise KeyError(f"This is not right, number of nodes must be divisible by 3")
        
        if self.graph_type == 'H2MIL' or self.graph_type == 'HiGT':
            from Benchmarking.utils.h2mil_graph import constrcut_h2mil_g as h2mil
            data = h2mil(data,path_to_dict, self.magnifications)
        elif self.graph_type == 'PatchGCN' or self.graph_type == 'DGCN':
            from Benchmarking.utils.patchgcn_graph import pt2graph
            #print(data)
            gap = int(data.ndata['x'].shape[0]/3)
            #print(self.magnifications)
            if self.magnifications[0] == '5x':
                data = pt2graph(data.ndata['x'][0:gap], data.ndata['c'][0:gap])
            elif self.magnifications[0] == '10x':
                #print(data.ndata['c'][gap:2*gap].shape, data.ndata['x'][gap:2*gap].shape)
                data = pt2graph(data.ndata['x'][gap:2*gap], data.ndata['c'][gap:2*gap])
            elif self.magnifications[0] == '20x':
                data = pt2graph(data.ndata['x'][2*gap:3*gap], data.ndata['c'][2*gap:3*gap])
            else:
                raise ValueError(f"Invalid magnification-there is no {self.magnifications[0]} as a key")
        else:
            pass #for now
        #print(data)
        label = self.label_dict[self.data_dict['label'][self.phase][idx]]
        #print('index', self.data_dict['label'][self.phase][idx] ,'label:', label)
        slide_name = self.data_dict['slide_id'][self.phase][idx]
        

        if self.transform:
            pass
        
        if self.super_patch:
            print(self.super_patch) # for now # doing some pooling stuff
        else:
            g = data


        return(g, label, slide_name)