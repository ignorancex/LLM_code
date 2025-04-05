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
import math


device = "cuda:0" if torch.cuda.is_available() else "cpu"



def main():
    
    apperances_threshold = 0.25
    #apperances_threshold = 0.3
    #apperances_threshold = 0.35
    #num_top_attributes = 1000
    #num_top_attributes = 50
    #num_top_attributes = 75
    #num_top_attributes = 1050
    #num_top_attributes = 300
    num_top_attributes = 4200

    print(f'get top {num_top_attributes} attribute class_label mutual information with cosine similarity threshold: {apperances_threshold}')


    with open(f"per_attr_total_mutual_info_threshold_{apperances_threshold}.json", "rb") as fp:
        attributes_mutual_info_dict = json.load(fp)

    sorted_attr_mutual_info_list = sorted(attributes_mutual_info_dict.items(), key=lambda item: item[1])
    top_attr_mutual_info_list = sorted_attr_mutual_info_list[-num_top_attributes:]
    print(f'len(top_attr_mutual_info_list): {len(top_attr_mutual_info_list)}, top_attr_mutual_info_list: \n{top_attr_mutual_info_list} ')

    top_attributes = [x[0] for x in top_attr_mutual_info_list]
    print(f'len(top_attributes): {len(top_attributes)}, top_attributes: \n{top_attributes} ')

    with open(f"top_{num_top_attributes}_attributes_threshold_{apperances_threshold}.json", "w") as fp:
        json.dump(top_attributes, fp)
  

if __name__ == "__main__":
    main()
