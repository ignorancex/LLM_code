import numpy as np
import constants as cst
import os
from torch.utils import data
import torch

  
def fi_2010_load(path, seq_size, horizon, all_features):
    dec_data = np.loadtxt(path + "/Train_Dst_NoAuction_ZScore_CF_7.txt")
    full_train = dec_data[:, :int(dec_data.shape[1] * cst.SPLIT_RATES[0])]
    full_val = dec_data[:, int(dec_data.shape[1] * cst.SPLIT_RATES[0]):]
    dec_test1 = np.loadtxt(path + '/Test_Dst_NoAuction_ZScore_CF_7.txt')
    dec_test2 = np.loadtxt(path + '/Test_Dst_NoAuction_ZScore_CF_8.txt')
    dec_test3 = np.loadtxt(path + '/Test_Dst_NoAuction_ZScore_CF_9.txt')
    full_test = np.hstack((dec_test1, dec_test2, dec_test3))
    
    if horizon == 1:
        tmp = 5
    elif horizon == 2:
        tmp = 4
    elif horizon == 3:
        tmp = 3
    elif horizon == 5:
        tmp = 2
    elif horizon == 10:
        tmp = 1
    else:
        raise ValueError("Horizon not found")
    
    train_labels = full_train[-tmp, :].flatten()
    val_labels = full_val[-tmp, :].flatten()
    test_labels = full_test[-tmp, :].flatten()
    
    train_labels = train_labels[seq_size-1:] - 1
    val_labels = val_labels[seq_size-1:] - 1
    test_labels = test_labels[seq_size-1:] - 1
    if all_features:
        train_input = full_train[:144, :].T
        val_input = full_val[:144, :].T
        test_input = full_test[:144, :].T
    else:
        train_input = full_train[:40, :].T
        val_input = full_val[:40, :].T
        test_input = full_test[:40, :].T
    train_input = torch.from_numpy(train_input).float()
    train_labels = torch.from_numpy(train_labels).long()
    val_input = torch.from_numpy(val_input).float()
    val_labels = torch.from_numpy(val_labels).long()
    test_input = torch.from_numpy(test_input).float()
    test_labels = torch.from_numpy(test_labels).long()
    return train_input, train_labels, val_input, val_labels, test_input, test_labels
    
    
