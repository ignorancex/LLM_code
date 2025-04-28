import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable
import torch.utils.data as Data
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import *
import numpy as np
import pickle
import argparse
import pandas as pd
#from ranking import *
from sklearn.preprocessing import StandardScaler
import tqdm
import fastai
import time
import copy
import matplotlib.pyplot as plt
import itertools
import random
import os

class Autoencoder(nn.Module):
    def __init__(self, representation_dim):
        super(Autoencoder, self).__init__()
        self.input_dim = representation_dim #2098
        self.layer1_dim = 1536
        self.layer2_dim = 1024
        self.layer3_dim = 768
        self.layer4_dim = 512

        #self.zdim = representation_dim
        
        activation_function = nn.Tanh()

        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, self.layer1_dim),
            activation_function,
            nn.Linear(self.layer1_dim, self.layer2_dim),
            activation_function,
            nn.Linear(self.layer2_dim, self.layer3_dim),
            activation_function,
        )

        self.encoder_mid = nn.Sequential(
            nn.Linear(self.layer3_dim, self.layer4_dim),
            activation_function
        )

        self.decoder = nn.Sequential(
            nn.Linear(self.layer4_dim, self.layer3_dim),
            activation_function,
            nn.Linear(self.layer3_dim, self.layer2_dim),
            activation_function,
            nn.Linear(self.layer2_dim, self.layer1_dim),
            activation_function,
            nn.Linear(self.layer1_dim, self.input_dim),
            activation_function
        ) 
        
    def forward(self, fused_rep):
        encoded = self.encoder(fused_rep)
        encoded_mid = self.encoder_mid(encoded)
        decoded = self.decoder(encoded_mid)
        return decoded,encoded_mid

def convert_dataframe_to_multi_col(representation_dataframe):
    entry = pd.DataFrame(representation_dataframe['Entry'])
    vector = pd.DataFrame(list(representation_dataframe['Vector']))
    multi_col_representation_vector = pd.merge(left=entry,right=vector,left_index=True, right_index=True)
    return multi_col_representation_vector

def convert_to_two_col(multi_col_representation_df):
    vals = multi_col_representation_df.iloc[:,1:(len(multi_col_representation_df.columns))]
    original_values_as_df = pd.DataFrame(columns=['Entry', 'Vector'])
    for index, row in tqdm.tqdm(vals.iterrows(), total = len(vals)):
        list_of_floats = [float(item) for item in list(row)]
        original_values_as_df.loc[index] = [multi_col_representation_df.iloc[index]['Entry']] + [list_of_floats]
    return original_values_as_df

def get_activation(name,activation):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

def train_model(model, train_loader, validation_loader, criterion,optimizer, num_epochs,\
                fused_tensors,device):
    since = time.time()

    train_loss_history = []
    val_loss_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 10**10
    
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min',verbose=True)
    loss_vals = []
    encoded_mid = None
    fused_tensors_size = fused_tensors.shape[1]
    for epoch in tqdm.tqdm(range(num_epochs)):

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
                inputs = train_loader
            else:
                model.eval()   # Set model to evaluate mode
                inputs = validation_loader

            loss = 0
            for batch_features in inputs:
               
                # reshape mini-batch data to [N, 784] matrix
                # load it to the active device
                fused_batch = fused_tensors[batch_features].view(-1, fused_tensors_size).to(device)

                # reset the gradients back to zero
                # PyTorch accumulates gradients on subsequent backward passes
                optimizer.zero_grad()
                
                decoded_rep, encoded_mid = model(fused_batch)

                # compute training reconstruction loss
                train_loss = criterion(decoded_rep, fused_batch)
                #train_loss = criterion(outputs, batch_features,target)

                # compute accumulated gradients
                train_loss.backward()

                # perform parameter update based on current gradients
                optimizer.step()

                # add the mini-batch training loss to epoch loss
                loss += train_loss.item()

            # compute the epoch training loss
            epoch_loss = loss / len(inputs)

            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_loss_history.append(epoch_loss)
                scheduler.step(epoch_loss)
            elif phase == 'train':
                train_loss_history.append(epoch_loss)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, train_loss_history,val_loss_history,loss_vals

def create_simple_ae(fused_rep_path):
    # Prepare fused representation vector
    fused_rep = pd.read_csv(fused_rep_path)
    scaler = StandardScaler()
    fused_rep.loc[:, fused_rep.columns != 'Entry'] = \
    scaler.fit_transform(fused_rep.loc[:, fused_rep.columns != 'Entry'])
    fused_rep_two_col = convert_to_two_col(fused_rep)
    fused_tensors = torch.tensor(list(fused_rep_two_col['Vector'].values))
    
    # Init training parameters
    batch_size = 128
    validation_split = .2
    shuffle_dataset = True
    seed = 42
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    # Creating data indices for training and validation splits:
    dataset_size = len(fused_rep_two_col)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset :
        np.random.seed(seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    train_loader = DataLoader(train_indices, batch_size=batch_size, 
                                                pin_memory=True,shuffle=True)
    validation_loader = DataLoader(val_indices, batch_size=batch_size,
                                                     pin_memory=True,shuffle=True)

    # Define Model
    representation_dim = len(fused_rep_two_col['Vector'][0])

    #  use gpu if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    epochs = 400
    # create a model from `AE` autoencoder class
    # load it to the specified device, either gpu or cpu
    model = Autoencoder(representation_dim).to(device)

    # create an optimizer object
    # Adam optimizer with learning rate 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    #optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    # mean-squared error loss
    criterion = nn.MSELoss()

    #Train Model
    best_model, train_loss_history,val_loss_history,loss_vals = train_model(model, train_loader, validation_loader, criterion, optimizer, epochs,fused_tensors,device)

    # create and register an forward hook
    activation = {}
    best_model.encoder_mid.register_forward_hook(get_activation('encoder_mid',activation))

    #Create simple_ae representation vectors
    simple_ae_rep = pd.DataFrame(columns=['Entry', 'Vector'])
    simple_ae_rep_size = fused_tensors.shape[1]
    best_model.eval()

    fused_tensors_size = fused_tensors.shape[1]
    with torch.no_grad():
        for index,row in tqdm.tqdm(fused_rep_two_col.iterrows(),total=len(fused_rep_two_col)):
            #fused_rep_tensor = torch.tensor(list(row['Vector']))
            fused_tensor = fused_tensors[index].view(-1, fused_tensors_size).to(device)
            
            _ = best_model(fused_tensor)
            coding_layer_output = activation['encoder_mid'].tolist()[0]    
            new_row = {'Entry':row['Entry'], 'Vector':coding_layer_output}
            simple_ae_rep = simple_ae_rep.append(new_row, ignore_index=True)

    simple_ae_multi_col = convert_dataframe_to_multi_col(simple_ae_rep )
    simple_ae_multi_col.to_csv("simple_ae.csv",index=False)


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python script.py param1")
    else:
        param1 = sys.argv[1]
        
        create_simple_ae(param1)
   # fused_rep_path = "/media/DATA2/sinem/hoper_lst/HOPER/case_study/case_study_results/modal_rep_ae_node2vec_binary_fused_representations_dataframe_multi_col.csv"
    
