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
dir = '../../HCP_3T_P/'
folds = 10
window_size = 30
max_window_size = 50
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
## Model parameters
batch_size = 1
################################################################

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

    # get all files 
    files = []
    for sub in test_sub: # loop through subjects id
        sub_files = glob.glob(dir+sub+'/*REST*_p.npy') # get the file names from the subfolder
        sub_files.sort() # sort the file names
        files += sub_files # add to the list

    # Load Model
    model = torch.load('new_models/win-30_transformer_fold_'+str(fold+1)+'_epo-10_win-30.pth')
    model.eval()
    test_mse = []
    test_regional_mse = []
    all_ses_data_predicted = []
    with torch.no_grad():
        progress = tqdm(range(len(files) * (1200 - max_window_size)))
        for f in files:
            ses_mse = []
            ses_regional_mse = []
            i = max_window_size
            f_data = np.load(f)
            data_for_test = f_data[0:max_window_size, :]
            while len(ses_mse) != (f_data.shape[0]-max_window_size):
                data = data_for_test[i-window_size:i, :]
                data = torch.tensor(data).unsqueeze(0)
                target = f_data[i, :]
                target = torch.tensor(target).unsqueeze(0)
                # prepare the inputs for the encoder and decoder
                encoder_input = data.to(device)
                decoder_input = data[:, -1, :].unsqueeze(1).to(device) # add one dimension for single time point
                # ensure the datatype is float64
                encoder_input = encoder_input.to(torch.float64)
                decoder_input = decoder_input.to(torch.float64)
                # Output of the Transformer
                pred = model(encoder_input, decoder_input) # (1, 1, # of regions)
                target = target.unsqueeze(1).to(device)
                target = target.to(torch.float64)
                error = mse_calc(target, pred)
                se = se_calc(target, pred)
                ses_mse.append(error.item())
                # calculate the regional MSE for each session 
                ses_regional_mse.append(se)
                # add the predicted timepoint to the time series
                pred = pred.squeeze() # (, 379)
                pred = pred.unsqueeze(0) # (1, 379)
                pred = pred.detach().cpu().numpy()
                data_for_test = np.concatenate((data_for_test, pred), axis=0)
                i += 1
                progress.update(1)
            test_mse.append(ses_mse)
            test_regional_mse.append(ses_regional_mse) 
            all_ses_data_predicted.append(data_for_test)
    progress.close() 

    test_mse = np.array(test_mse)
    test_regional_mse = np.array(test_regional_mse)
    all_ses_data_predicted = np.array(all_ses_data_predicted)
    print(test_mse.shape)
    print(test_regional_mse.shape)
    print(all_ses_data_predicted.shape)
    np.save('new_models/test/all_sub_fold_'+str(fold+1)+'_test_mse_entire_pred.npy', test_mse)
    # np.save('new_models/test/fold_'+str(fold+1)+'_test_regional_se_entire_pred.npy', test_regional_mse)
    np.save('new_models/test/all_sub_fold_'+str(fold+1)+'_fmri_predicted.npy', all_ses_data_predicted)
