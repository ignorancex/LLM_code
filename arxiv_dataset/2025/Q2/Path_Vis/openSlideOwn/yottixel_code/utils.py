from os.path import join
import ast

import numpy as np
import pandas as pd
from cv2 import filter2D
import bitarray
from bitarray import util as butil


class BoB:
    def __init__(self, barcodes, file_name, site, diagnosis):
        self.barcodes = [bitarray.bitarray(b.tolist()) for b in barcodes]
        self.file_name = file_name
        self.site = site
        self.diagnosis = diagnosis
        
    def select_subset(self, n=3):
        idx = np.arange(len(self.barcodes))
        np.random.shuffle(idx)
        idx = idx[:n]
        return BoB(barcodes=[self.barcodes[i] for i in idx])
    
    def distance(self, bob):
        total_dist = []
        for feat in self.barcodes:
            distances = [butil.count_xor(feat, b) for b in bob.barcodes]
            total_dist.append(np.min(distances))
        retval = np.median(total_dist)
        return retval
        

def RGB2HSD(X):
    eps = np.finfo(float).eps
    X[np.where(X==0.0)] = eps
    
    OD = -np.log(X / 1.0)
    D  = np.mean(OD,3)
    D[np.where(D==0.0)] = eps
    
    cx = OD[:,:,:,0] / (D) - 1.0
    cy = (OD[:,:,:,1]-OD[:,:,:,2]) / (np.sqrt(3.0)*D)
    
    D = np.expand_dims(D,3)
    cx = np.expand_dims(cx,3)
    cy = np.expand_dims(cy,3)
            
    X_HSD = np.concatenate((D,cx,cy),3)
    return X_HSD


def clean_thumbnail(thumbnail):
    thumbnail_arr = np.asarray(thumbnail)
    # writable thumbnail
    wthumbnail = np.zeros_like(thumbnail_arr)
    wthumbnail[:, :, :] = thumbnail_arr[:, :, :]
    # Remove pen marking here
    # We are skipping this
    # This  section sets regoins with white spectrum as the backgroud regoin
    thumbnail_std = np.std(wthumbnail, axis=2)
    wthumbnail[thumbnail_std<5] = (np.ones((1,3), dtype="uint8")*255)
    thumbnail_HSD = RGB2HSD( np.array([wthumbnail.astype('float32')/255.]) )[0]
    kernel = np.ones((30,30),np.float32)/900
    thumbnail_HSD_mean = filter2D(thumbnail_HSD[:,:,2],-1,kernel)
    wthumbnail[thumbnail_HSD_mean<0.05] = (np.ones((1,3),dtype="uint8")*255)
    return wthumbnail


def save_mosaic(mosaic, patch_save_dir):
    df = pd.DataFrame(mosaic)
    df['loc'] = df['loc'].apply(lambda x: str(x))
    df['wsi_loc'] = df['wsi_loc'].apply(lambda x: str(x))
    df['rgb_histogram'] = df['rgb_histogram'].apply(lambda x: str(x.tolist()))  # Convert numpy array to list before converting to string
    df.to_hdf(join(patch_save_dir, 'mosaic.h5'), key='df', mode='w')


def load_mosaic(mosaic_path):
    df = pd.read_hdf(mosaic_path, 'df')
    # Convert back to original forms
    df['loc'] = df['loc'].apply(ast.literal_eval)
    df['wsi_loc'] = df['wsi_loc'].apply(ast.literal_eval)
    df['rgb_histogram'] = df['rgb_histogram'].apply(lambda x: np.array(ast.literal_eval(x)))
    return df.to_dict('records')
    