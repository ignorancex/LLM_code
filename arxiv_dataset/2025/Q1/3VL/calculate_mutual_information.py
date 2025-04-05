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

def calc_mutual(joint_occur, x1_margin_occur, x2_margin_occur, total):
    if joint_occur == 0:
        return 0


    px1_2 = joint_occur / total
    px1 = x1_margin_occur / total
    px2 = x2_margin_occur / total
    log_input = px1 * px2
    log_input = px1_2 / log_input
    log_term = math.log2(log_input)
    res = px1_2 * log_term
    #print(f'calc_mutual: joint_occur: {joint_occur}, x1_margin_occur: {x1_margin_occur}, x2_margin_occur: {x2_margin_occur}, total: {total}')
    #print(f'px1_2: {px1_2}, px1: {px1}, px2: {px2}, log_input: {log_input}, log_term: {log_term}, res: {log_term}')
    return res

def main():
    
    with open('full_adjectives.json', 'rb') as fp:
        attributes = json.load(fp)

    with open(f"image_net_class_label_counter.json", "rb") as fp:
        class_label_counter = json.load(fp)

    #apperances_threshold = 0.25
    apperances_threshold = 0.3
    #apperances_threshold = 0.35
    print(f'calculating attribute class_label mutual information with cosine similarity threshold: {apperances_threshold}')

    with open(f"attribute_num_appearances_threshold_{apperances_threshold}.json", "rb") as fp:
        attribute_num_appearances = json.load(fp)
        
    with open(f"num_attr_appearances_per_class_threshold_{apperances_threshold}.json", "rb") as fp:
        attribute_num_per_class = json.load(fp)

    total_num_samples = 1281167
    per_attr_total_mutual_info = dict()
    per_attr_per_class_mutual_info = dict()
    for attr in tqdm(attributes):
        attr_marginal_occurence = attribute_num_appearances[attr]
        not_attr_marginal_occurence = total_num_samples - attr_marginal_occurence
        attr_marginal_dist = attr_marginal_occurence / total_num_samples
        if attr_marginal_dist in [0, 1] :
            per_attr_total_mutual_info[attr] = 0 
            #not needed in per_attr_per_class_mutual_info dict
            continue
        not_attr_marginal_dist = 1 - attr_marginal_dist
        total_attr_mutual_info = 0
        per_class_mutual_info = dict()
        for class_label in range(1000):
            
            class_str = str(class_label)
            #count_per_class = Counter(attribute_num_per_class[attr])
            class_marginal_occurence = class_label_counter[class_str]
            class_marginal_dist = class_marginal_occurence / total_num_samples
            #has_attr_joint_occurence = count_per_class[class_label]
            has_attr_joint_occurence = attribute_num_per_class[attr][class_str] if class_str in attribute_num_per_class[attr] else 0
            not_attr_joint_occurence = class_marginal_occurence - has_attr_joint_occurence
            has_attr_joint_dist = has_attr_joint_occurence / total_num_samples
            not_attr_joint_dist = not_attr_joint_occurence / total_num_samples

            class_mutual_info = 0

            has_attr_info = calc_mutual(has_attr_joint_occurence, attr_marginal_occurence, class_marginal_occurence, total_num_samples)
            class_mutual_info += has_attr_info

            not_attr_info = calc_mutual(not_attr_joint_occurence, not_attr_marginal_occurence, class_marginal_occurence, total_num_samples)
            class_mutual_info += not_attr_info
            #print(f'attr: {attr}, class_label: {class_label}, class_mutual_info: {class_mutual_info}')

            #if has_attr_joint_dist > 0:
            #    log_input_denominator = attr_marginal_dist*class_marginal_dist
            #    if log_input_denominator > 0:
            #        log_input = has_attr_joint_dist / log_input_denominator
            #        log_term = math.log2(log_input)
            #        class_mutual_info += has_attr_joint_dist*log_term

            #if not_attr_joint_dist > 0:
            #    log_input_denominator = not_attr_marginal_dist*class_marginal_dist
            #    if log_input_denominator > 0:
            #        log_input = not_attr_joint_dist / log_input_denominator
            #        log_term = math.log2(log_input)
            #        class_mutual_info += not_attr_joint_dist*log_term

            per_class_mutual_info[str(class_label)] = class_mutual_info
            #print(f'class_str: {str(class_label)}, \nper_class_mutual_info: \n{per_class_mutual_info}')
            total_attr_mutual_info += class_mutual_info

        per_attr_total_mutual_info[attr] = total_attr_mutual_info
        per_attr_total_mutual_info = {k: v for k, v in sorted(per_attr_total_mutual_info.items(), key=lambda item: item[1])}
        per_attr_per_class_mutual_info[attr] = per_class_mutual_info
          
    with open(f"per_attr_total_mutual_info_threshold_{apperances_threshold}.json", "w") as fp:
        json.dump(per_attr_total_mutual_info, fp)

    with open(f"per_attr_per_class_mutual_info_threshold_{apperances_threshold}.json", "w") as fp:
        json.dump(per_attr_per_class_mutual_info, fp)

    



if __name__ == "__main__":
    main()
