import numpy as np

import torch
from torch.utils.data import Dataset

from torch import nn
import NoInterNet_model as NIN

device = 'cuda' if torch.cuda.is_available() else 'cpu'

#utils

#Range defined in NIN

#Preprocessing

def load_Pkf(file_name, fraction_file, k_max=0.8, k_min=0, log=True, multipole=0, norm_fs=False, norm_c=False, multipole_correction=False, P2_only=False, n_load=None):
    import pandas as pd

    if multipole_correction:
        assert multipole > 0, "Cannoct correct quadrupole if it is not an input, set multipole = 2, 4"

    Pk_paths = pd.read_csv(file_name)

    if 'f' in Pk_paths.columns:
        fs = Pk_paths['f']
    else:
        print("Attention! It is not advise to get the fraction from a separate file! Fishy things may happen!")
        fs = np.loadtxt(fraction_file)

    Pks_int = []
    labels = []

    N = len(Pk_paths.index) if n_load is None else n_load

    for i, row in Pk_paths.head(N).iterrows():
        Pk_int  = np.loadtxt(row['Pk_int'])
        Pk_true = np.loadtxt(row['Pk_true'])

        assert np.allclose(Pk_int[:,0], Pk_true[:,0]), f"{i:d}: True and contaminated Pks have different binning!"

        selk = (Pk_int[:,0] > k_min) & (Pk_int[:,0] < k_max)

        if log:
            if multipole == 0:
                Pks_int.append(np.log10(Pk_int[:,1][selk]))
            elif multipole == 2:
                Pks_int.append(np.log10(np.concatenate((Pk_int[:,1][selk], Pk_int[:,2][selk]))))
            
            label = np.log10(Pk_true[:,1][selk] / Pk_int[:,1][selk])
        else:
            if multipole == 0:
                Pks_int.append(Pk_int[:,1][selk])
            elif multipole == 2:
                Pks_int.append(np.concatenate((Pk_int[:,1][selk], Pk_int[:,2][selk])))

            #labels orderd as [correction(k), f]
            if multipole_correction:
                Pt = np.concatenate((Pk_true[:,1][selk], Pk_true[:,2][selk])) #true P0+P2
                Pi = np.concatenate((Pk_int[:,1][selk], Pk_int[:,2][selk])) #contaminated P0+P2
                label = Pt / Pi #P0+P2 correction
            elif P2_only:
                label = Pk_true[:,2][selk] / Pk_int[:,2][selk] #correction only for P2
            else:
                label = Pk_true[:,1][selk] / Pk_int[:,1][selk] #correction only for P0
        f = minmax(fs[i]) if norm_fs else fs[i]
        label = np.append(label, f)
        labels.append(label)

    labels = np.array(labels)
    if norm_c:
        max_corr = labels[:,:-1].max(axis=0)
        min_corr = labels[:,:-1].min(axis=0)
        labels[:,:-1] = maxmin_corr(labels[:,:-1], max_corr, min_corr)

    return np.array(Pks_int), labels, sum(selk)

def load_Pkf_feat(file_name, fraction_file, k_max=0.8, k_min=0, log=True, multipole=0, norm_fs=False, norm_c=False):
    import pandas as pd

    Pk_paths = pd.read_csv(file_name)
    
    if 'f' in Pk_paths.columns:
        fs = Pk_paths['f']
    else:
        fs = np.loadtxt(fraction_file)

    Pks_int = []
    labels = []

    for i, row in Pk_paths.iterrows():
        Pk_int  = np.loadtxt(row['Pk_int'])
        Pk_true = np.loadtxt(row['Pk_true'])

        assert np.allclose(Pk_int[:,0], Pk_true[:,0]), f"{i:d}: True and contaminated Pks have different binning!"

        selk = (Pk_int[:,0] > k_min) & (Pk_int[:,0] < k_max)

        f = minmax(fs[i]) if norm_fs else fs[i]
        if log:
            if multipole == 0:
                feat = np.append(np.log10(Pk_int[:,1][selk]), f)
                Pks_int.append(feat)
            elif multipole == 2:
                Pks_int.append(np.log10(np.concatenate((Pk_int[:,1][selk], Pk_int[:,2][selk]))))
            
            label = np.log10(Pk_true[:,1][selk] / Pk_int[:,1][selk])
        else:
            if multipole == 0:
                feat = np.append(Pk_int[:,1][selk], f)
                Pks_int.append(feat)
            elif multipole == 2:
                Pks_int.append(np.concatenate((Pk_int[:,1][selk], Pk_int[:,2][selk])))

            #labels orderd as [correction(k), f]
            label = Pk_true[:,1][selk] / Pk_int[:,1][selk]
        
        labels.append(label)

    labels = np.array(labels)
    if norm_c:
        max_corr = labels.max(axis=0)
        min_corr = labels.min(axis=0)
        labels = maxmin_corr(labels, max_corr, min_corr)

    return np.array(Pks_int), labels, sum(selk)

def load_PBf(file_name, fraction_file, k_max=0.8, k_min=0, log=True, kf=6.2832e-03, norm_fs=False, norm_c=False, n_load=None):
    import pandas as pd

    PB_paths = pd.read_csv(file_name)
    
    if 'f' in PB_paths.columns:
        fs = PB_paths['f']
    else:
        fs = np.loadtxt(fraction_file)

    PBs_int = []
    labels = []

    N = len(PB_paths.index) if n_load is None else n_load

    for i, row in PB_paths.head(N).iterrows():
        Pk_int  = np.loadtxt(row['Pk_int'])
        Bk_int  = np.loadtxt(row['Bk_int'])
        Pk_true = np.loadtxt(row['Pk_true'])

        assert np.allclose(Pk_int[:,0], Pk_true[:,0]), f"{i:d}: True and contaminated Pks have different binning!"

        selk = (Pk_int[:,0] > k_min) & (Pk_int[:,0] < k_max)
        selB = (Bk_int[:,2] * kf > k_min) & (Bk_int[:,0] * kf < k_max) #k3<=k2<=k1

        if log:
            PB_int = np.concatenate((np.log10(Pk_int[:,1][selk]), np.log10(Bk_int[:,6][selB])))
            PBs_int.append(PB_int)
            
            label = np.log10(Pk_true[:,1][selk] / Pk_int[:,1][selk])
        
        else:
            PB_int = np.concatenate((Pk_int[:,1][selk], Bk_int[:,6][selB]))
            PBs_int.append(PB_int)
            

        #labels orderd as [correction(k), f]
        label = Pk_true[:,1][selk] / Pk_int[:,1][selk]
        f = minmax(fs[i]) if norm_fs else fs[i]
        label = np.append(label, f)
        labels.append(label)

    labels = np.array(labels)
    if norm_c:
        max_corr = labels[:,:-1].max(axis=0)
        min_corr = labels[:,:-1].min(axis=0)
        labels[:,:-1] = maxmin_corr(labels[:,:-1], max_corr, min_corr)

    return np.array(PBs_int), labels, sum(selk), sum(selB)

def load_PBf_B(file_name, fraction_file, k_max=0.8, k_min=0, log=True, kf=6.2832e-03, norm_fs=False, norm_c=False, n_load=None):
    import pandas as pd

    PB_paths = pd.read_csv(file_name)
    
    if 'f' in PB_paths.columns:
        fs = PB_paths['f']
    else:
        fs = np.loadtxt(fraction_file)

    PBs_int = []
    labels = []

    N = len(PB_paths.index) if n_load is None else n_load

    for i, row in PB_paths.head(N).iterrows():
        Pk_int  = np.loadtxt(row['Pk_int'])
        Bk_int  = np.loadtxt(row['Bk_int'])
        Bk_true = np.loadtxt(row['Bk_true'])

        assert np.allclose(Bk_int[:,0], Bk_true[:,0]), f"{i:d}: True and contaminated Pks have different binning!"

        selk = (Pk_int[:,0] > k_min) & (Pk_int[:,0] < k_max)
        selB = (Bk_int[:,2] * kf > k_min) & (Bk_int[:,0] * kf < k_max) #k3<=k2<=k1

        if log:
            PB_int = np.concatenate((np.log10(Pk_int[:,1][selk]), np.log10(Bk_int[:,6][selB])))
            PBs_int.append(PB_int)
            
            label = np.log10(Bk_true[:,1][selB] / Bk_int[:,1][selB])
        
        else:
            PB_int = np.concatenate((Pk_int[:,1][selk], Bk_int[:,6][selB]))
            PBs_int.append(PB_int)
            

        #labels orderd as [correction(k), f]
        label = Bk_true[:,1][selB] / Bk_int[:,1][selB]
        f = minmax(fs[i]) if norm_fs else fs[i]
        label = np.append(label, f)
        labels.append(label)

    labels = np.array(labels)
    if norm_c:
        max_corr = labels[:,:-1].max(axis=0)
        min_corr = labels[:,:-1].min(axis=0)
        labels[:,:-1] = maxmin_corr(labels[:,:-1], max_corr, min_corr)

    return np.array(PBs_int), labels, sum(selk), sum(selB)

#normalize P defined in NIN

def minmax(f):
    f -= 0.01
    f /= (0.11 - 0.01)
    return f

def inv_minmax(f):
    f *= (0.11 - 0.01)
    f += 0.01
    return f

def maxmin_corr(corr, max, min):
    corr -= min
    corr /= (max - min)
    return corr

def inv_maxmin_corr(corr, max, min):
    corr *= (max - min)
    corr += min
    return corr

def normalize_PB(PBs, nP):
    '''Normalise each Pk and Bk by theit max'''
    normP = np.max(PBs[:,:nP], axis=1)
    normB = np.max(PBs[:,nP:], axis=1)

    PBs[:,:nP] /= normP[:,np.newaxis]
    PBs[:,nP:] /= normB[:,np.newaxis]

    return PBs, normP, normB

#PkDataset defined in NIN

class PkDataset_norm(Dataset):
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

        Pk_int = np.append(Pk_int, torch.log10(norm))
        
        return Pk_int, label, norm
    
class PBDataset(Dataset):
    def __init__(self, PBs_int, labels, normP, transform=torch.from_numpy, target_transform=torch.from_numpy):
        self.PBs_int = PBs_int
        self.labels = labels
        self.normP = normP
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        PB_int = self.PBs_int[idx]
        label = self.labels[idx]
        normP = self.normP[idx]
        
        if self.transform:
            PB_int = self.transform(PB_int).type(torch.float32)
        if self.target_transform:
            label = self.target_transform(label).type(torch.float32)
            normP = self.target_transform(normP).type(torch.float32)
        
        return PB_int, label, normP

#custom loss functions

class LogFractionLoss(nn.Module):
    def __init__(self):
        super(LogFractionLoss, self).__init__()

    def forward(self, output, target):
        '''loss = log(MSE(correction)) + log(MSE(f)).
           It assumes the following ordering of the labels [correction(k), f]'''
        criterion = nn.MSELoss()
        
        loss_correction = criterion(output[:,:-1], target[:,:-1])
        loss_f = criterion(output[:,-1], target[:,-1])

        loss = torch.log(loss_correction) + torch.log(loss_f)

        return loss
    
class LogMomentFractionLoss(nn.Module):
    def __init__(self, lenk):
        super(LogMomentFractionLoss, self).__init__()
        self.lenk = lenk

    def forward(self, output, target):
        '''loss = log(MSE(correction)) + log(MSE(f)) + log(MSE(sigma_correction)) + log(MSE(sigma_f)).
           It assumes the following ordering of the labels [correction(k), f, sigma_correction(k), sigma_f]'''
        criterion = nn.MSELoss()
        lenk = self.lenk

        loss_correction_primary = criterion(output[:,:lenk], target[:,:lenk])
        loss_f_primary = criterion(output[:,lenk], target[:,lenk])

        loss_correction_secondary = criterion(output[:,lenk+1:2*lenk+1]**2, (target[:,:lenk] - output[:,:lenk])**2)
        loss_f_secondary = criterion(output[:,-1]**2, (target[:,lenk] - output[:,lenk])**2)

        loss = torch.log(loss_correction_primary) + torch.log(loss_correction_secondary) + torch.log(loss_f_primary) + torch.log(loss_f_secondary)

        return loss
    
class LogMomentLoss(nn.Module):
    def __init__(self, lenk):
        super(LogMomentLoss, self).__init__()
        self.lenk = lenk

    def forward(self, output, target):
        '''loss = log(MSE(correction)) + log(MSE(sigma_correction)).
           It assumes the following ordering of the labels [correction(k), sigma_correction(k)]'''
        criterion = nn.MSELoss()
        lenk = self.lenk

        loss_correction_primary = criterion(output[:,:lenk], target[:,:lenk])
        loss_correction_secondary = criterion(output[:,lenk:]**2, (target[:,:lenk] - output[:,:lenk])**2)

        loss = torch.log(loss_correction_primary) + torch.log(loss_correction_secondary)

        return loss

#Networks

class NoInterNet_fraction_compress(nn.Module):
    def __init__(self, input_size, n_in, output_size=None, n_out=None, n_min=8):
        '''Find correction and interloper fraction
        ---Inputs---
        output_size : int
            size of the correction array'''
        
        super().__init__()
                
        self.input_size  = input_size
        self.output_size = input_size if output_size is None else output_size
        self.output_size += 1
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
    
class NoInterNet_fraction_compress_norm(nn.Module):
    def __init__(self, input_size, n_in, output_size=None, n_out=None, n_min=8):
        '''Find correction and interloper fraction
        ---Inputs---
        output_size : int
            size of the correction array'''
        
        super().__init__()
                
        self.input_size  = input_size + 1
        self.output_size = input_size if output_size is None else output_size
        self.output_size += 1
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
    

class NoInterNet_fraction_compress_inference(nn.Module):
    def __init__(self, input_size, n_in, output_size=None, n_out=None, n_min=8):
        '''Find first and second moment of correction and interloper fraction
        ---Inputs---
        output_size : int
            size of the correction array'''
        
        super().__init__()
                
        self.input_size  = input_size
        self.output_size = input_size * 2 if output_size is None else output_size * 2 
        self.output_size += 2
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
    
class NoInterNet_fraction_compress_knowf_inference(nn.Module):
    def __init__(self, input_size, n_in, output_size=None, n_out=None, n_min=8):
        '''Find first and second moment of correction
        ---Inputs---
        output_size : int
            size of the correction array'''
        
        super().__init__()
                
        self.input_size  = input_size + 1
        self.output_size = input_size * 2 if output_size is None else output_size *2 #not sure of the else outcome
        print(self.input_size, self.output_size)

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
    
#Saving and loading

#checkpoint definded in NIN
#resume definded in NIN

#Training

def train_loop_inference(dataloader, model, optimizer, lenk):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    current_loss = 0.0
    for batch, (X, y, n) in enumerate(dataloader):
        X = X.to(device)
        y = y.to(device)
        pred = model(X)

        corr_pred = pred[:,:lenk]
        corr_labl = y[:,:lenk]
        f_pred = pred[:,lenk]
        f_labl = y[:,lenk]
        corr_std = pred[:,lenk+1:lenk*2+1]
        f_std = pred[:,-1]

        loss_correction_primary = torch.mean(torch.sum((corr_pred - corr_labl)**2, axis=1), axis=0)
        loss_f_primary = torch.mean((f_pred - f_labl)**2, axis=0)

        loss_correction_secondary = torch.mean(torch.sum(((corr_pred - corr_labl)**2 - corr_std**2)**2, axis=1), axis=0)
        loss_f_secondary = torch.mean(((f_pred - f_labl)**2 - f_std**2)**2, axis=0)

        loss = torch.log(loss_correction_primary) + torch.log(loss_correction_secondary) + torch.log(loss_f_primary) + torch.log(loss_f_secondary)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        current_loss += loss.item()
        
        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            
    return current_loss / num_batches
            
def test_loop_inference(dataloader, model, lenk):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss = 0
    
    with torch.no_grad():
        for X, y, n in dataloader:
            X = X.to(device)
            y = y.to(device)
            pred = model(X)

            corr_pred = pred[:,:lenk]
            corr_labl = y[:,:lenk]
            f_pred = pred[:,lenk]
            f_labl = y[:,lenk]
            corr_std = pred[:,lenk+1:lenk*2+1]
            f_std = pred[:,-1]
    
            loss_correction_primary = torch.mean(torch.sum((corr_pred - corr_labl)**2, axis=1), axis=0)
            loss_f_primary = torch.mean((f_pred - f_labl)**2, axis=0)
    
            loss_correction_secondary = torch.mean(torch.sum(((corr_pred - corr_labl)**2 - corr_std**2)**2, axis=1), axis=0)
            loss_f_secondary = torch.mean(((f_pred - f_labl)**2 - f_std**2)**2, axis=0)
    
            loss = torch.log(loss_correction_primary) + torch.log(loss_correction_secondary) + torch.log(loss_f_primary) + torch.log(loss_f_secondary)

            test_loss += loss.item()
            
    test_loss /= num_batches
    print(f"Test Error: \n Avg loss: {test_loss:>8f} \n")
    
    return test_loss
    
def training_inference(train_dataloader, val_dataloader, model, optimizer, epochs, patience, output, lenk, neurons=60):
    
    best_loss = 1e5
    best_epoch = -1
    val_history = []
    train_history = []
    
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loss = train_loop_inference(train_dataloader, model, optimizer, lenk)
        val_loss = test_loop_inference(val_dataloader, model, lenk)
        train_history.append(train_loss)
        val_history.append(val_loss)
        
        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = t
            NIN.checkpoint(model, output+f"best_model-{neurons}.pth")
            print("New best model")
        elif t - best_epoch > patience:
            print(f"Early stopped training at epochs {t+1}")
            break
            
    print("Done!")
    
    return train_history, val_history

def train_loop_inference_knowf(dataloader, model, optimizer, lenk):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    current_loss = 0.0
    for batch, (X, y, n) in enumerate(dataloader):
        X = X.to(device)
        y = y.to(device)
        pred = model(X)

        corr_pred = pred[:,:lenk]
        corr_labl = y[:,:lenk]
        corr_std = pred[:,lenk:]

        loss_correction_primary = torch.mean(torch.sum((corr_pred - corr_labl)**2, axis=1), axis=0)

        loss_correction_secondary = torch.mean(torch.sum(((corr_pred - corr_labl)**2 - corr_std**2)**2, axis=1), axis=0)
        
        loss = torch.log(loss_correction_primary) + torch.log(loss_correction_secondary)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        current_loss += loss.item()
        
        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            
    return current_loss / num_batches

def test_loop_inference_knowf(dataloader, model, lenk):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss = 0
    
    with torch.no_grad():
        for X, y, n in dataloader:
            X = X.to(device)
            y = y.to(device)
            pred = model(X)

            corr_pred = pred[:,:lenk]
            corr_labl = y[:,:lenk]
            corr_std = pred[:,lenk:]
    
            loss_correction_primary = torch.mean(torch.sum((corr_pred - corr_labl)**2, axis=1), axis=0)
    
            loss_correction_secondary = torch.mean(torch.sum(((corr_pred - corr_labl)**2 - corr_std**2)**2, axis=1), axis=0)
            
            loss = torch.log(loss_correction_primary) + torch.log(loss_correction_secondary)
            test_loss += loss.item()
            
    test_loss /= num_batches
    print(f"Test Error: \n Avg loss: {test_loss:>8f} \n")
    
    return test_loss
    
def training_inference_knowf(train_dataloader, val_dataloader, model, optimizer, epochs, patience, output, lenk, neurons=60):
    
    best_loss = 1e5
    best_epoch = -1
    val_history = []
    train_history = []
    
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loss = train_loop_inference_knowf(train_dataloader, model, optimizer, lenk)
        val_loss = test_loop_inference_knowf(val_dataloader, model, lenk)
        train_history.append(train_loss)
        val_history.append(val_loss)
        
        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = t
            NIN.checkpoint(model, output+f"best_model-{neurons}.pth")
            print("New best model")
        elif t - best_epoch > patience:
            print(f"Early stopped training at epochs {t+1}")
            break
            
    print("Done!")
    
    return train_history, val_history