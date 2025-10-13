import torch
import os
import glob
import numpy as np
import pandas as pd
import scipy
import dgl
from dgl.data.utils import save_graphs
from scipy import sparse
import h5py

class graph_construction(object):

    def __init__(self, config):
        self.mags = [str(i)+'x' for i in sorted(config.mags)]
        self.feat_location = config.feat_location
        self.graph_location = config.graph_location
        self.manifest_location = config.manifest_location
    
    def get_slide_bulks(self):
        extentions = ['pt','pth', 'bin', 'h5']
        feat_path = {}
        feat_path= []
        for ext in extentions:
            path_wildcard = os.path.join(self.feat_location, '**', '*.' + ext)
            feat_path.extend(glob.glob(path_wildcard, recursive=True))
        feat_path = sorted(feat_path)

        if len(feat_path)==0:
            raise ValueError("Wrong Directory Bro! Listen to me: I told you that your features must be stored either as .pt, .pth, h5 or .bin formats!!!!!!!!! Do not mess with me :/")

        return feat_path
    
    
    def get_slide_info(self):
        #add patient info from manifest blahb blha
        manifest = pd.read_csv(self.manifest_location)
        slide_info = {}

        for index in range(len(manifest.slide_id)):
            slide_info[manifest.slide_id[index]] = [manifest.patient_id[index], manifest.subtype[index]]
        
        return slide_info

    def load_multi_mags(self, slide_path):
        path_to_dict =slide_path
        if path_to_dict.split('.')[-1] == 'h5':
            f = h5py.File(path_to_dict, 'r')
            data = {}
            coor = {}
            for key in self.mags:
                data[key] = torch.tensor(f['features'][key][:])
                coor[key] = torch.tensor(f['coords'][key][:])
            f = None

        elif path_to_dict.split('.')[-1] == 'pt':
            data = torch.load(path_to_dict)['features']
            coor = torch.load(path_to_dict)['coords']
        else:
            raise NotImplementedError(f"we do not support this format {path_to_dict.split('.')[-1]}")
        feats = {'features': data, 'coords': coor} 
        mag_0 = list(feats['features'].keys())[0]
        num_feats = len(feats['features'][mag_0])
        slide_name = slide_path.split('/')[-1].split('.')[0]
        for mag in feats['features'].keys():
            if len(feats['features'][mag]) != len(feats['features'][mag_0]):
                print('sick:', slide_name)
                raise ValueError("GOD Bless you bro! You are indeeed a genious!! Go check your patches, you have not extracted equal numbers of patches from each magnification")
        
        
        subtype = slide_path.split('/')[-2]
        
        return feats, slide_name, num_feats, subtype
    
    def build_graph(self, feats, slide_name, num_feats, subtype):
        n = num_feats #:(
        print("n: ", n)
        
        for mag in self.mags:
            if not mag in feats['features'].keys():
                raise ValueError(f"Are you kidding me?!! the list of magnification you gave me is not the same as the magnifications that you extracted patches with!! This happend for {mag}!")

        n_mag = len(feats['features'])
        A = np.zeros((n_mag*n,n_mag*n), dtype=np.float16)

        for block_row in range(n_mag):
            for block_col in range(n_mag):
                if block_row == block_col:
                    A[block_row*n:(block_row+1)*n, block_col*n:(block_col+1)*n] = np.ones((n,n), dtype=np.float16)
                if abs(block_row-block_col)==1:
                    A[block_row*n:(block_row+1)*n, block_col*n:(block_col+1)*n] = np.diag(np.ones((1,n), dtype=np.float16).reshape(n,))
        
        sA = sparse.csr_matrix(A)
        g_ = dgl.from_scipy(sA)
        g = g_.int()
        #print(g.idtype)
        tensor_list = list(feats['features'].values())
        coords_list = list(feats['coords'].values())
        #print(tensor_list)
        x = torch.cat(tensor_list, dim=0)
        c = torch.cat(coords_list, dim=0)
        if len(x.shape) > 2:
            g.ndata['x'] = x[:,12,:]
            g.ndata['c'] = c[:,12,:]
        else:
            g.ndata['x'] = x
            g.ndata['c'] = c
        # g.edata['w'] = no need at this point
        path_to_save = os.path.join(self.graph_location, subtype)
        try:
            os.makedirs(path_to_save, exist_ok = True)
        except OSError as error:
            print("Directory '%s' can not be created")
        
        save_graphs(path_to_save + '/' + slide_name +'.bin', [g])
        print(slide_name + '---Done! ')

    def run(self):
        feat_path = self.get_slide_bulks()
        slide_info = self.get_slide_info()
        for slide_path in feat_path:
            feats, slide_name, num_feats, subtype = self.load_multi_mags(slide_path)
            print(slide_name, flush=True)
            #if not slide_name in slide_info.keys():
            #    raise ValueError("This slide has not been found in your manifest file!")
            #if slide_info[slide_name][1] != subtype:
            #    raise ValueError(f"The subtype assigned to the features does not match the subtype assigned to the slide in the manifest file! This error is happening for {slide_name} ...
            #    with {slide_info[slide_name][1]} as the manifest's subtype and {subtype} as the features subtype"
            #)
            try:
                self.build_graph(feats, slide_name, num_feats, subtype)
            except:
                pass
        
        print('Well, I\'m doooooooooooonnnnnne brooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo!')

    