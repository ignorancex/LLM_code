import os
import glob
import matplotlib.pyplot as plt # plotting library
import numpy as np # this module is useful to work with numerical arrays
import pandas as pd
import random
import torch
from torch.utils.data import DataLoader,random_split
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from fmri_dataset import rfMRIDataset
from model import *
from tqdm import tqdm
import pickle

def mse_calc(x, x_hat):
    reproduction_loss = nn.functional.mse_loss(x_hat, x)
    return reproduction_loss

def se_calc(x, x_hat):
    x1 = x.cpu()
    x2 = x_hat.cpu()
    x1 = x1.numpy()
    x2 = x2.numpy()
    se = (x1 - x2)**2
    return se

################################################################
""" parameters """
folds = 10
dir = '../../HCP_3T_P/'
max_window_size = 50
window_size = 30
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
shuffle = False
batch_size = 1
################################################################
if shuffle:
    print('Shuffle the input time series')
# get all subjects id
all_sub = os.listdir(dir)
all_sub.sort()

# 10-fold
with open('test_subjects.pickle', 'rb') as file:
    test_sub_split = pickle.load(file)
for fold in range(folds):
    test_sub = test_sub_split[fold]
    print(f'Fold {fold+1}')
    print(test_sub)
    print(len(test_sub))

    # load the data
    test_data = rfMRIDataset(dir, test_sub, window_size, max_window_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False, pin_memory=True)
    # test
    model = torch.load('new_models/transformer_fold_'+str(fold+1)+'_epo-10_win-30.pth')
    model.eval()
    test_mse = []
    test_regional_mse = []

    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(tqdm(test_dataloader)):
            if shuffle:
                data = data.squeeze()
                idx = torch.randperm(data.size(0))
                data = data[idx, :]
                data = data.unsqueeze(0)

            # prepare the inputs for the encoder and decoder
            encoder_input = data.to(device)
            decoder_input = data[:, -1, :].unsqueeze(1).to(device) # add one dimension for single time point
            # ensure the datatype is float64
            encoder_input = encoder_input.to(torch.float64)
            decoder_input = decoder_input.to(torch.float64)
            # Output of the Transformer
            pred = model(encoder_input, decoder_input) # (batch_size, 1, # of regions)
            target = target.unsqueeze(1).to(device)
            target = target.to(torch.float64)
            error = mse_calc(target, pred)
            # se = se_calc(target, pred)
            test_mse.append(error.item())
            # calculate the regional MSE for each session 
            # test_regional_mse.append(se)
            
    test_mse = np.array(test_mse)
    # test_regional_mse = np.array(test_regional_mse)
    print(test_mse.shape)
    # print(test_regional_mse.shape)
    if shuffle:
        np.save('new_models/test/all_sub_fold_'+str(fold+1)+'_test_mse_shuffled.npy', test_mse)
        # np.save('large_window/test/fold_'+str(fold+1)+'_test_regional_se_shuffled.npy', test_regional_mse)
    else:
        np.save('new_models/test/all_sub_fold_'+str(fold+1)+'_test_mse.npy', test_mse)
        # np.save('large_window/test/fold_'+str(fold+1)+'_test_regional_se.npy', test_regional_mse)
