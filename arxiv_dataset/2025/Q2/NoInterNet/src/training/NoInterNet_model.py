import numpy as np

import torch
from torch.utils.data import Dataset

from torch import nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'

#utils
class Range(object):
    def __init__(self, start, end):
        self.start = start
        self.end = end

    def __eq__(self, other):
        return self.start <= other <= self.end

    def __contains__(self, item):
        return self.__eq__(item)

    def __iter__(self):
        yield self

    def __str__(self):
        return '[{0},{1}]'.format(self.start, self.end)

#Preprocessing
def load_Pk(file_name, k_max=0.8, log=True, multipole=0):
    import pandas as pd

    Pk_paths = pd.read_csv(file_name)

    Pks_int = []
    labels = []

    for i, row in Pk_paths.iterrows():
        Pk_int  = np.loadtxt(row['Pk_int'])
        Pk_true = np.loadtxt(row['Pk_true'])

        assert np.allclose(Pk_int[:,0], Pk_true[:,0]), f"{i:d}: True and contaminated Pks have different binning!"

        selk = Pk_int[:,0]  < k_max

        if log:
            if multipole == 0:
                Pks_int.append(np.log10(Pk_int[:,1][selk]))
            elif multipole == 2:
                Pks_int.append(np.log10(np.concatenate((Pk_int[:,1][selk], Pk_int[:,2][selk]))))
            
            label = np.log10(Pk_true[:,1][selk] / Pk_int[:,1][selk])
            labels.append(label)
        else:
            if multipole == 0:
                Pks_int.append(Pk_int[:,1][selk])
            elif multipole == 2:
                Pks_int.append(np.concatenate((Pk_int[:,1][selk], Pk_int[:,2][selk])))

            label = Pk_true[:,1][selk] / Pk_int[:,1][selk]
            labels.append(label)

    return np.array(Pks_int), np.array(labels), sum(selk)

def normalize_Pk(Pks):
    '''Normalise each Pk by its max'''
    norm = np.max(Pks, axis=1)

    return Pks / norm[:,np.newaxis], norm

def normalize_02(Pks, nP):
    '''Normalize each P0 by its max and each P2 by their max'''
    norm0 = np.max(Pks[:,:nP], axis=1)
    norm2 = np.max(Pks[:,nP:], axis=1)

    Pks[:,:nP] /= norm0[:,np.newaxis]
    Pks[:,nP:] /= norm2[:,np.newaxis]

    return Pks, norm0, norm2

class PkDataset(Dataset):
    def __init__(self, Pks_int, labels, norm, transform=torch.from_numpy, target_transform=torch.from_numpy):
        self.Pks_int = Pks_int
        self.labels = labels
        self.norm = norm
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        Pk_int = self.Pks_int[idx]
        label = self.labels[idx]
        norm = self.norm[idx]
        
        if self.transform:
            Pk_int = self.transform(Pk_int).type(torch.float32)
        if self.target_transform:
            label = self.target_transform(label).type(torch.float32)
            norm = self.target_transform(norm).type(torch.float32)
        
        return Pk_int, label, norm

#custom loss functions

class LogMSELoss(nn.Module):
    def __init__(self):
        super(LogMSELoss, self).__init__()

    def forward(self, output, target):
        nk = output.shape[1]
        criterion = nn.MSELoss()
        loss = 0
        for i in range(nk):
            loss += torch.log(criterion(output[:,i], target[:,i]))
        
        return loss
    
#Networks
class NoInterNet_compress(nn.Module):
    def __init__(self, input_size, n_in, output_size=None, n_out=None, n_min=8):
        super().__init__()
        self.input_size  = input_size
        self.output_size = input_size if output_size is None else output_size
        modules = [nn.Linear(self.input_size, n_in),
                   nn.LeakyReLU()] #input
         
        self.n_hidden = 0 #number of hidden layers
        n_check = n_in
        n_out = n_in if n_out is None else n_out
        while n_check > n_min:
            self.n_hidden += 1
            n_out_hid = int(n_check / 2)
            print(self.n_hidden, n_check, n_out_hid)
            modules.append(nn.Linear(n_check, n_out_hid))
            modules.append(nn.LeakyReLU()) #L compress
            n_check = n_out_hid
        while n_check < n_out:
            self.n_hidden += 1
            n_out_hid  = n_check * 2
            print(self.n_hidden, n_check, n_out_hid)
            modules.append(nn.Linear(n_check, n_out_hid))
            modules.append(nn.LeakyReLU()) #L decompress
            n_check = n_out_hid
            
        modules.append(nn.Linear(n_check,self.output_size)) #output
        
        self.linear_relu_stack = nn.Sequential(*modules)
        
    def forward(self, x):
        #x = self.flatten(x)
        out = self.linear_relu_stack(x)
        return out
    
class NoInterNet_fraction_compress_knowf(nn.Module):
    def __init__(self, input_size, n_in, output_size=None, n_out=None, n_min=8):
        '''Find first and second moment of correction and interloper fraction
        ---Inputs---
        output_size : int
            size of the correction array'''
        
        super().__init__()
                
        self.input_size  = input_size + 1
        self.output_size = input_size if output_size is None else output_size
        modules = [nn.Linear(self.input_size, n_in),
                   nn.LeakyReLU()] #input
         
        self.n_hidden = 0 #number of hidden layers
        n_check = n_in
        n_out = n_in if n_out is None else n_out
        while n_check > n_min:
            self.n_hidden += 1
            n_out_hid = int(n_check / 2)
            print(self.n_hidden, n_check, n_out_hid)
            modules.append(nn.Linear(n_check, n_out_hid))
            modules.append(nn.LeakyReLU()) #L compress
            n_check = n_out_hid
        while n_check < n_out:
            self.n_hidden += 1
            n_out_hid  = n_check * 2
            print(self.n_hidden, n_check, n_out_hid)
            modules.append(nn.Linear(n_check, n_out_hid))
            modules.append(nn.LeakyReLU()) #L decompress
            n_check = n_out_hid
            
        modules.append(nn.Linear(n_check,self.output_size)) #output
        
        self.linear_relu_stack = nn.Sequential(*modules)
        
    def forward(self, x):
        out = self.linear_relu_stack(x)
        return out
    
#Saving and loading
def checkpoint(model, name):
    torch.save(model.state_dict(), name)
    
def resume(model, name, device):
    model.load_state_dict(torch.load(name, map_location=torch.device(device)))

#Training
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    current_loss = 0.0
    for batch, (X, y, n) in enumerate(dataloader):
        X = X.to(device)
        y = y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        current_loss += loss.item()
        
        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            
    return current_loss / num_batches
            
def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss = 0
    
    with torch.no_grad():
        for X, y, n in dataloader:
            X = X.to(device)
            y = y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            
    test_loss /= num_batches
    print(f"Test Error: \n Avg loss: {test_loss:>8f} \n")
    
    return test_loss
    
def training(train_dataloader, val_dataloader, model, loss_fn, optimizer, epochs, patience, output, neurons=60):
    
    best_loss = 1e5
    best_epoch = -1
    val_history = []
    train_history = []
    
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loss = train_loop(train_dataloader, model, loss_fn, optimizer)
        val_loss = test_loop(val_dataloader, model, loss_fn)
        train_history.append(train_loss)
        val_history.append(val_loss)
        
        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = t
            checkpoint(model, output+f"best_model-{neurons}.pth")
            print("New best model")
        elif t - best_epoch > patience:
            print(f"Early stopped training at epochs {t+1}")
            break
            
    print("Done!")
    
    return train_history, val_history
