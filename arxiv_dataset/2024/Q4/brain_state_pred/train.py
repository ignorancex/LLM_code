import os
import glob
import numpy as np
import torch
from fmri_dataset import rfMRIDataset
from model import *
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

import pandas as pd
from torch import nn
import torch.nn.functional as F
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

############################################################
dir = '../../HCP_3T_P/'
window_size = 30
max_window_size = 50
## Model parameters
dim_val = 760 # This can be any value divisible by n_heads. 512 is used in the original transformer paper.
n_heads = 8 # The number of attention heads (aka parallel attention layers). dim_val must be divisible by this number
n_decoder_layers = 4 # Number of times the decoder layer is stacked in the decoder
n_encoder_layers = 4 # Number of times the encoder layer is stacked in the encoder
input_size = 379 # The number of input variables. 1 if univariate forecasting.
dec_seq_len = 1 # length of input given to decoder. Can have any integer value.
output_sequence_length = 1 # Length of the target sequence, i.e. how many time steps should your forecast cover
num_predicted_features = 379
batch_first = True
## Training parameters
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
lr = 1e-4
epochs = 10
batch_size = 512
shuffle = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
############################################################


# k-fold XV
num_folds = 10
with open('test_subjects.pickle', 'rb') as file:
    test_sub_split = pickle.load(file)
with open('train_subjects.pickle', 'rb') as file:
    train_sub_split = pickle.load(file)
fold = 0
for fold in range(num_folds):
    train_sub = train_sub_split[fold]
    test_sub = test_sub_split[fold]
    
    print(f"Fold {fold+1}:")
    print(f"Train subjects length: {len(train_sub)}")
    print(f"Test subjects length: {len(test_sub)}")
    enc_seq_len = window_size # length of input given to encoder. Can have any integer value.
    max_seq_len = enc_seq_len # What's the longest sequence the model will encounter? Used to make the positional encoder
    print(f'Fold {fold+1}: Train the model with window size = {window_size} and {epochs} epochs.')
    train_data = rfMRIDataset(dir, train_sub, window_size, max_window_size)
    train_dataloader = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=shuffle)
    # validation
    test_data = rfMRIDataset(dir, test_sub, window_size, max_window_size)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False, pin_memory=True)

    # initialize the model
    model = TimeSeriesTransformer(
        dim_val=dim_val,
        input_size=input_size,
        n_heads=n_heads,
        dec_seq_len=dec_seq_len,
        max_seq_len=max_seq_len,
        out_seq_len=output_sequence_length,
        n_decoder_layers=n_decoder_layers,
        n_encoder_layers=n_encoder_layers,
        batch_first=batch_first,
        num_predicted_features=num_predicted_features)
    model.to(device)
    model = model.double()

    # train the model
    # Define the loss function and optimizer
    loss_func = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # training
    loss_hist = []
    val_loss_hist = []
    for epoch in range(epochs):
        model.train()
        single_epo_loss = []
        progress_bar = tqdm(range(len(train_dataloader)))
        for batch_idx, (data, target) in enumerate(train_dataloader):

            encoder_input = data.to(device)
            decoder_input = data[:, -1, :].unsqueeze(1).to(device) # add one dimension for single time point
            encoder_input = encoder_input.to(torch.float64)
            decoder_input = decoder_input.to(torch.float64)
            pred = model(encoder_input, decoder_input) # (batch_size, 1, # of regions)
            trg = target.unsqueeze(1).to(device)
            trg = trg.to(torch.float64)
            loss = loss_func(trg, pred)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Storing the losses in a list for plotting
            single_epo_loss.append(loss.cpu().detach().numpy())
            progress_bar.update(1)
        progress_bar.close()
        single_epo_loss = np.array(single_epo_loss)
        loss_hist.append(single_epo_loss)

        # validation
        model.eval()
        test_mse = []
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(tqdm(test_dataloader)):
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
                se = se_calc(target, pred)
                test_mse.append(error.item())
        test_mse = np.array(test_mse)
        val_loss_hist.append(test_mse)
        print("Epoch", epoch + 1, "complete!", "\tAverage Loss(MSE): ", np.mean(single_epo_loss), 'Validation Loss (MSE): ', np.mean(test_mse))
    loss_hist = np.array(loss_hist)
    val_loss_hist = np.array(val_loss_hist)

    # save the model
    torch.save(model, 'transformer_fold_'+str(fold+1)+'_epo-'+str(epochs)+'_win-'+str(window_size)+'.pth')
    # save the loss history
    np.save('transformer_train_loss_fold_'+str(fold+1)+'_epo-'+str(epochs)+'_win-'+str(window_size)+'.npy', loss_hist)
    np.save('transformer_valid_loss_fold_'+str(fold+1)+'_epo-'+str(epochs)+'_win-'+str(window_size)+'.npy', val_loss_hist)
    # plot the loss history
    x_values = list(range(1, epochs+1))
    plt.plot(x_values, loss_hist.mean(axis=1))
    plt.xlabel('Epochs')
    plt.ylabel('MSE Loss')
    plt.title('Training Loss')
    plt.savefig('transformer_train_loss_fold_'+str(fold+1)+'_epo-'+str(epochs)+'_win-'+str(window_size)+'.png')
    plt.clf()
