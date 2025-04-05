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
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.tree import export_text
from sklearn.metrics import accuracy_score


device = "cuda:0" if torch.cuda.is_available() else "cpu"


def get_data_and_labels(csv_file_name, num_features = 512):
    
    print(f'get_data_and_labels from csv_file_name: {csv_file_name}')
    dataset = pd.read_csv(csv_file_name)
    #print(f'\ndataset: \n{dataset}')
    #print(f'\ndataset.info(): \n')
    #dataset.info()
    #print(f'\ndataset.head(): \n{dataset.head()}')
    
    #X = dataset.values[:, 0:num_features]
    X = dataset.drop(columns="label")
    feature_names = X.columns
    #Y = dataset.values[:, num_features]
    Y = dataset["label"].astype('int16')

    #print(f'X.shape: {X.shape}, Y.shape: {Y.shape}')
    #print(f'Y[0:5]: {Y[0:5]}')
    #print(f'\nX[0:5]: \n{X[0:5]}')
    #print(f'\nfeature_names: \n{feature_names}')

    return X, Y, feature_names



def main():
    
    #train_filename = 'clip_imagenet_train.csv'
    #train_dataset = pd.read_csv(train_filename)
    #train_dataset.head()

    num_features = 512
    #X_train = train_dataset[:, 0:num_features]
    #Y_train = train_dataset[:, num_features]

    X_train, Y_train, feature_names = get_data_and_labels('clip_imagenet_train.csv', num_features)
    X_test, Y_test, _ = get_data_and_labels('clip_imagenet_test.csv', num_features)

    #X_temp, Y_temp, feature_names = get_data_and_labels('tmp_train.csv', num_features)
    
    clf = DecisionTreeClassifier(max_depth =100, random_state = 42)

    clf.fit(X_train, Y_train)
    preds = clf.predict(X_test)
    #clf.fit(X_temp, Y_temp)
    #preds = clf.predict(X_temp)

    accuracy = accuracy_score(Y_test, preds)
    
    tree_rules = export_text(clf, feature_names = list(feature_names))

    #print(f'\ntree is: \n{tree_rules}')
    print(f"\nDecisionTreeClassifier accuracy on ImageNet test set: {accuracy*100:.2f}%")

if __name__ == "__main__":
    main()
