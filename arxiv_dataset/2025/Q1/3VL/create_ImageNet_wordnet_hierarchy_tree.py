# Imports
#pip install transformers
#pip install ftfy regex tqdm
#pip install git+https://github.com/openai/CLIP.git

import os
import torch
import torchvision 
import torchvision.datasets as datasets  
import torchvision.transforms as transforms  
from torch import optim  
from torch import nn  
from torch.utils.data import DataLoader  
from tqdm import tqdm
import clip  
from torchvision.datasets import CIFAR100
from torchvision.datasets import ImageNet
from tqdm import tqdm

import pandas as pd
import numpy as np
#from sklearn.datasets import load_iris
#from sklearn.tree import DecisionTreeClassifier
#from sklearn import tree
from sklearn.tree import export_text
from sklearn.metrics import accuracy_score
import json
#import _pickle as cPickle
import pickle
from collections import Counter
import sys
from robustness.robustness.tools.imagenet_helpers import ImageNetHierarchy


device = "cuda:0" if torch.cuda.is_available() else "cpu"

def get_data_and_labels(csv_file_name, num_features = 512, subset_size=-1):
    
    print(f'get_data_and_labels from csv_file_name: {csv_file_name}')
    df = pd.read_csv(csv_file_name)
    #print(f'\ndataset: \n{dataset}')
    #print(f'\ndataset.info(): \n')
    #dataset.info()
    #print(f'\ndataset.head(): \n{dataset.head()}')
    dataset = df.to_numpy()
    #X = dataset.values[:, 0:num_features]
    
    if subset_size != -1:
        np.random.shuffle(dataset)
        dataset = dataset[0:subset_size]

    #X = dataset.drop(columns="label")
    #feature_names = X.columns
    #Y = dataset.values[:, num_features]
    #Y = dataset["label"].astype('int16')

    X = dataset[:,:-1]
    Y = dataset[:,-1].astype('int16')


    print(f'X.shape: {X.shape}, Y.shape: {Y.shape}')
    #print(f'Y[0:5]: {Y[0:5]}')
    #print(f'\nX[0:5]: \n{X[0:5]}')
    #print(f'\nfeature_names: \n{feature_names}')

    #return X, Y, feature_names
    return X, Y



def main():
    
    image_net_path = "/disk5/dataset/ImageNet"
    image_net_info_path = "/disk5/dataset/ImageNet/imagenet_info"
    image_net_hier = ImageNetHierarchy(image_net_path, image_net_info_path)

    #for wnid in image_net_hier.in_wnids:
    #        node = image_net_hier.tree[wnid]
    #        node.children = set()

    #for wnid in image_net_hier.in_wnids:
    #        node = image_net_hier.tree[wnid]
    #        while node.parent_wnid is not None:
    #            image_net_hier.tree[node.parent_wnid].children.update(node.wnid)
    #            node = image_net_hier.tree[node.parent_wnid]

    for cnt, (wnid, ndesc_in, ndesc_total) in enumerate(image_net_hier.wnid_sorted):
    
        if cnt < 3:
            print(f"WordNet ID: {wnid}, Name: {image_net_hier.wnid_to_name[wnid]}, #ImageNet descendants: {ndesc_in}")
            print(f"children WordNet IDs: {image_net_hier.tree[wnid].children}")
        

        

        



if __name__ == "__main__":
    main()
