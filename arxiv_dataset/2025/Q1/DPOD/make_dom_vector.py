import json
import numpy as np 
import clip 
import dataset_mismatch  
import math
import clip_classifier
from custom_collate import collate_mismatch
import dataset_mismatch  
import json 
import os 
import torch
import torchvision
import torch.nn as nn
import numpy as np 
from torch.utils.data import DataLoader
import argparse
import io
import clip_classifier
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import time 
import clip 
from sklearn.metrics import confusion_matrix, classification_report
from PIL import Image
import torch
from numpy import dot
from numpy.linalg import norm
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def cos(a,b):
    a = np.squeeze(a)
    b = np.squeeze(b)    
    cos_sim = dot(a, b)/(norm(a)*norm(b))
    return cos_sim
def load_img_pil(image_path):
    with open(image_path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')
model_settings = {'pdrop': 0.05}
device= "cuda:0"
base_clip, preprocess = clip.load("ViT-B/32", device=device)

pol_img_embeddings = []
pol_text_embeddings = []
non_pol_img_embeddings = []
non_pol_text_embeddings = []

data_paths_all_nc_train = json.load(open(""))# path to the json file where you wish to find and store the domain vector


domain_embeddings = {}   #key is the domain and value is a list of joing embeddings
dom_idx = {}
dom_idx_iter = 0 
all_domains = []
final_dom_embeddings = {}
for idx in  data_paths_all_nc_train:
    all_domains.append( data_paths_all_nc_train[idx]['topic'])
all_domains = list(set(all_domains))

for domain in all_domains:
    dom_idx[str(domain)] = dom_idx_iter
    dom_idx_iter +=1
dom_all_embeddings = {}


for domain in all_domains:
    dom_all_embeddings[domain] = []


for k,v in data_paths_all_nc_train.items():
    print("Number completed: ", k)

    caption = v['caption']
    caption_tokenized = clip.tokenize(caption).to(device)
    image_path = v['image_path']
    pil_img = load_img_pil(image_path)

    transform_img = preprocess(pil_img)

    transform_img =transform_img.unsqueeze(0).to(device)

    encode_img = base_clip.encode_image(transform_img)
    encode_text = base_clip.encode_text(caption_tokenized)
    encoded_img = encode_img / encode_img.norm(dim=-1, keepdim=True) 
    encoded_text = encode_text / encode_text.norm(dim=-1, keepdim=True)
    """
    Arrays kept in cpu for memory usage purpose
    """
    
    encoded_img = encoded_img.cpu().detach().numpy()
    encoded_text = encoded_text.cpu().detach().numpy()
    encode_img = encode_img.cpu().detach().numpy()
    encode_text = encode_text.cpu().detach().numpy()
    
    joint_embedding = encode_img*encode_text
    dom_all_embeddings[v['topic']].append(joint_embedding)
for k,v in dom_all_embeddings.items():

    dom_all_embeddings[k] = np.mean(v,axis=0)
for k,v in dom_all_embeddings.items():
    dom_vector_embeddings = []
    for i in range(0,54):
        dom_vector_embeddings.append(0) 
    for d,e in dom_all_embeddings.items():
        cos_sim = cos(v,e) 
        if cos_sim< 1:
            cos_sim = cos_sim


        
        dom_vector_embeddings[dom_idx[d]] = cos_sim
    final_dom_embeddings[k] = dom_vector_embeddings

data_path_all_train = json.load(open(""))
for idx in data_path_all_train:
    data_path_all_train[idx]['dom_embed'] = np.array(final_dom_embeddings[data_path_all_train[idx]['topic']],dtype = np.float64).tolist()

        
        
        
        
        
        
