# Script used to tensorize trained NN models

import os
import sys
import getopt
import json
from importlib import util

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torchvision
import torchaudio

import tensorkrowch as tk
from tensorkrowch.decompositions import tt_rss
from tensorkrowch.utils import random_unitary

torch.set_num_threads(1)

cwd = os.getcwd()
p_english_list = [0.005, 0.01, 0.05,
                 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
                 0.95, 0.99, 0.995]
out_rate = 1000


def import_file(full_name, path):
    """Returns a python module given its path"""
    spec = util.spec_from_file_location(full_name, path)
    mod = util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


#################
# Load Datasets #
#################
# MARK: Load Datasets

class CustomCommonVoice(Dataset):
    """
    Class for the (imbalanced) datasets created.

    Parameters
    ----------
    p_english : float (p_english_list)
        Proportion of audios of people with english accent in the dataset.
    idx : int [0, 9]
        Index of the annotations to be used. For each ``p_english`` there are 10
        datasets.
    set : str
        Indicates which dataset is to be loaded.
    transform : torchvision.transforms
        Transformations of the dataset (data augmentation, normalization, etc.)
    target_transform : func
        Transformation of the target attribute (not used).
    """
    
    def __init__(self,
                 p_english,
                 idx,
                 set="train_df.tsv",
                 transform=None):
        
        global p_english_list
        if (p_english not in p_english_list) or ((idx < 0) or (idx > 9)):
            raise ValueError(
                f'`p_english` can only take values within {p_english_list}, '
                f'and `idx` should be between 0 and 9')
        
        global cwd
        self.dataset = torchaudio.datasets.COMMONVOICE(
            root=os.path.join(cwd, 'CommonVoice'),
            tsv=os.path.join('datasets', str(p_english), str(idx), set))
        self.transform = transform

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        x, y, z = self.dataset[index]
        if self.transform:
            x = self.transform((x, y))
        return x, int(z['sex'])
    

def resample(x):
    global out_rate
    x, in_rate = x
    resample_trans = torchaudio.transforms.Resample(in_rate, out_rate)
    return resample_trans(x)

def crop(x):
    global out_rate
    llimit = (x.size(1) // 2 - out_rate // 2)
    rlimit = (x.size(1) // 2 + out_rate // 2)
    x = x[:, llimit:rlimit].flatten()
    if x.size(0) < out_rate:
        return None
    return x

def rfft(x):
    if x is None:
        return None
    return torch.fft.rfft(x)[:-1].abs()

def normalize(x):
    x = x / 200
    x = torch.where(x <= 0, 1e-5, x)
    x = torch.where(x >= 1, 1 - 1e-5, x)
    return x

transform = torchvision.transforms.Compose([
    torchvision.transforms.Lambda(resample),
    torchvision.transforms.Lambda(crop),
    torchvision.transforms.Lambda(rfft),
    torchvision.transforms.Lambda(normalize)
    ])

def none_collate(batch):
    batch = list(filter(lambda x: x[0] is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


def load_data(p_english, idx, batch_size):
    """Loads dataset performing the required transformations for train or test."""
    
    # Load datasets
    global transform
    train_dataset = CustomCommonVoice(p_english,
                                      idx,
                                      set="train_df.tsv",
                                      transform=transform)
    val_dataset = CustomCommonVoice(p_english,
                                    idx,
                                    set="val_df.tsv",
                                    transform=transform)
    test_dataset = CustomCommonVoice(p_english,
                                     idx,
                                     set="test_df.tsv",
                                     transform=transform)
    
    # Create DataLoaders
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              collate_fn=none_collate,
                              shuffle=True)
    val_loader = DataLoader(val_dataset,
                            batch_size=batch_size,
                            collate_fn=none_collate,
                            shuffle=False)
    test_loader = DataLoader(test_dataset,
                             batch_size=batch_size,
                             collate_fn=none_collate,
                             shuffle=False)
    
    return train_loader, val_loader, test_loader


def load_sketch_samples(p_english, idx, batch_size):
    """Loads sketch samples to tensorize models."""
    
    # Load datasets
    global transform
    test_tensorize_dataset = CustomCommonVoice(p_english,
                                               idx,
                                               set="test_df_tensorize.tsv",
                                               transform=transform)
    test_unused_dataset = CustomCommonVoice(p_english,
                                            idx,
                                            set="test_df_unused.tsv",
                                            transform=transform)
    
    # Create DataLoaders
    test_tensorize_loader = DataLoader(test_tensorize_dataset,
                                       batch_size=500,
                                       collate_fn=none_collate,
                                       shuffle=False)
    test_unused_loader = DataLoader(test_unused_dataset,
                                    batch_size=batch_size,
                                    collate_fn=none_collate,
                                    shuffle=False)
    
    return test_tensorize_loader, test_unused_loader


def training_epoch_tn(device, model, embedding, criterion, optimizer,
                      train_loader, logs, n_batches=None):
    if n_batches is not None:
        n_batches = min(n_batches, len(train_loader))
    else:
        n_batches = len(train_loader)
    i = 0
    
    running_loss = 0
    running_acc = 0

    model.train()
    for data, labels in train_loader:
        data = data.to(device)
        labels = labels.to(device)
        
        # Forward
        scores = model(embedding(data),
                       inline_input=False,
                       inline_mats=False)
        scores = scores.pow(2)
        scores = scores / scores.norm(dim=1, keepdim=True)
        scores = torch.where(scores == 0, 1e-10, scores)
        scores = scores.log()
        
        loss = criterion(scores, labels)
        
        with torch.no_grad():
            _, preds = torch.max(scores, 1)
            accuracy = (preds == labels).float().mean().item()
            running_acc += accuracy
            running_loss += loss.item()
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient descent
        optimizer.step()
        
        i += 1
        if i >= n_batches:
            break
    
    logs['train_losses'].append(running_loss / n_batches)
    logs['train_accs'].append(running_acc / n_batches)
    
    return model, logs


def test_tn(device, model, embedding, criterion, test_loader,
            logs, n_batches=None):
    """Computes accuracy on test set."""
    running_loss = 0
    running_acc = 0
    
    if n_batches is not None:
        n_batches = min(n_batches, len(test_loader))
    else:
        n_batches = len(test_loader)
    i = 0
    
    model.eval()
    with torch.no_grad():
        for data, labels in test_loader:
            data = data.to(device)
            labels = labels.to(device)
            
            scores = model(embedding(data),
                           inline_input=False,
                           inline_mats=False)
            scores = scores.pow(2)
            scores = scores / scores.norm(dim=1, keepdim=True)
            scores = torch.where(scores == 0, 1e-10, scores)
            scores = scores.log()
            
            loss = criterion(scores, labels)
            
            _, preds = torch.max(scores, 1)
            accuracy = (preds == labels).float().mean().item()
            running_acc += accuracy
            running_loss += loss.item()
            
            i += 1
            if i >= n_batches:
                break
    
    logs['val_losses'].append(running_loss / n_batches)
    logs['val_accs'].append(running_acc / n_batches)
    
    return logs


###############################################################################
###############################################################################

#############
# Tensorize #
#############
# MARK: Tensorize

def tensorize_retrain(model_class, n_epochs, p_english, start_idx, end_idx):
    """Tensorizes models trained with the given ``p_english`` and ``idx``."""
    global p_english_list
    
    device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')
    
    if p_english is not None:
        p_english_list_aux = [p_english]
    else:
        p_english_list_aux = p_english_list
        
    if start_idx is not None:
        idx_range_aux = range(start_idx, end_idx)
    else:
        idx_range_aux = range(10)
    
    # Check tuned config of balanced model
    config_dir = os.path.join(cwd, 'results', '0_train_nns',
                              model_class.name, '0.5')
    with open(os.path.join(config_dir, 'tuned_config.json'), 'r') as f:
        config = json.load(f)
    
    softmax = nn.Softmax(dim=1)
    criterion = nn.NLLLoss()
        
    for p_english in p_english_list_aux:
        for idx in idx_range_aux:
            models_dir = os.path.join(cwd, 'results', '0_train_nns',
                                      model_class.name, str(p_english), str(idx))
            
            sketches_dir = os.path.join(cwd, 'results', '6_privacy',
                                        model_class.name, 'sketches',
                                        str(p_english), str(idx))
            os.makedirs(sketches_dir, exist_ok=True)
            
            cores_dir = os.path.join(cwd, 'results', '6_privacy',
                                     model_class.name, 'cores',
                                     str(p_english), str(idx))
            os.makedirs(cores_dir, exist_ok=True)
            
            recores_dir = os.path.join(cwd, 'results', '6_privacy',
                                       model_class.name, 'recores',
                                       str(p_english), str(idx))
            os.makedirs(recores_dir, exist_ok=True)
            
            priv_recores_dir = os.path.join(cwd, 'results', '6_privacy',
                                            model_class.name, 'priv_recores',
                                            str(p_english), str(idx))
            os.makedirs(priv_recores_dir, exist_ok=True)
            
            n_models = len(os.listdir(models_dir))
            n_cores = len(os.listdir(cores_dir))
            
            for s in range(n_cores, n_models):
                # Load state_dict
                state_dict_dir = list(filter(lambda f: f.startswith(f'{s}_'),
                                             os.listdir(models_dir)))[0]
                state_dict = torch.load(os.path.join(models_dir, state_dict_dir),
                                        weights_only=False)
                
                # Initialize model with balanced config
                model = model_class(config)
                model.load_state_dict(state_dict)
                model.eval()
                model.to(device)
                
                def fn(samples):
                    aux = softmax(model(samples))
                    aux = torch.where(aux > 0.5, 1., 0.)
                    return aux
                
                # Load data
                batch_size_loader = 64
                tensorize_loader, unused_loader = load_sketch_samples(
                    p_english, idx, batch_size_loader)
                sketch_samples, sketch_labels = next(iter(tensorize_loader))
                perm = torch.randperm(sketch_samples.size(0))
                sketch_samples = sketch_samples[perm]
                sketch_labels = sketch_labels[perm]
                
                # Save sketch samples
                torch.save((sketch_samples, sketch_labels),
                           os.path.join(sketches_dir, f'{s}.pt'))
                
                n_features = out_rate // 2 + 1
                
                # Tensorization hyperprameters
                embed_dim = 2
                bond_dim = 5
                cum_percentage = 0.99
                domain_multiplier = 1
                
                domain_dim = domain_multiplier * embed_dim
                domain = torch.arange(domain_dim).float() / domain_dim

                n_samples = 100
                batch_size_tensorize = 1000

                def embedding(x):
                    x = tk.embeddings.poly(x, degree=embed_dim - 1)
                    return x
                
                # Tensorize
                cores = tt_rss(function=fn,
                               embedding=embedding,
                               sketch_samples=sketch_samples[:n_samples],
                               labels=sketch_labels[:n_samples],
                               domain=domain,
                               domain_multiplier=domain_multiplier,
                               rank=bond_dim,
                               cum_percentage=cum_percentage,
                               batch_size=batch_size_tensorize,
                               device=device,
                               verbose=False)
                
                # Test MPS model
                tn_model = tk.models.MPSLayer(tensors=cores)
                tn_model.to(device)
                
                tn_model.trace(
                    torch.zeros(1, n_features - 1, embed_dim).to(device),
                    inline_input=False,
                    inline_mats=False
                )
                
                sketch_dataset = TensorDataset(sketch_samples[:n_samples],
                                               sketch_labels[:n_samples])
                sketch_loader = DataLoader(sketch_dataset,
                                           batch_size=n_samples,
                                           collate_fn=none_collate,
                                           shuffle=False)
                
                logs = {'train_losses': [],
                        'val_losses': [],
                        'train_accs': [],
                        'val_accs': []}
                
                # On test dataset
                logs = test_tn(device=device,
                               model=tn_model,
                               embedding=embedding,
                               criterion=criterion,
                               test_loader=unused_loader,
                               logs=logs,
                               #n_batches=5
                               )
                test_acc = logs['val_accs'][-1]
                
                # On sketch samples
                logs = test_tn(device=device,
                               model=tn_model,
                               embedding=embedding,
                               criterion=criterion,
                               test_loader=sketch_loader,
                               logs=logs,
                               #n_batches=5
                               )
                sketch_acc = logs['val_accs'][-1]
                
                print(f'**{model_class.name}** (p: {p_english}, i: {idx}, '
                      f's: {s}) => '
                      f'Test Acc.: {test_acc:.4f}, '
                      f'Sketch Acc.: {sketch_acc:.4f}')
                
                torch.save(
                    cores,
                    os.path.join(cores_dir,
                                 f'{s}_{test_acc:.4f}_{sketch_acc:.4f}.pt')
                )
                
                # Retrain
                optimizer = torch.optim.Adam(tn_model.parameters(),
                                             lr=1e-4,
                                             weight_decay=1e-6)
                best_val_acc = test_acc
                best_cores = [t.detach().clone()
                              for t in tn_model.tensors]
                
                for epoch in range(n_epochs):
                    tn_model, logs = training_epoch_tn(
                        device=device,
                        model=tn_model,
                        embedding=embedding,
                        criterion=criterion,
                        optimizer=optimizer,
                        train_loader=tensorize_loader,
                        logs=logs,
                        # n_batches=10,
                        )
                    
                    # Validate
                    logs = test_tn(device=device,
                                   model=tn_model,
                                   embedding=embedding,
                                   criterion=criterion,
                                   test_loader=unused_loader,
                                   logs=logs,
                                   n_batches=10,
                                   )
                    
                    print(f'**{model_class.name}** (p: {p_english}, i: {idx}, '
                          f's: {s}) => '
                          f'Epoch: {epoch + 1}/{n_epochs}, '
                          f'Train Loss: {logs["train_losses"][-1]:.3f}, '
                          f'Val Loss: {logs["val_losses"][-1]:.3f}, '
                          f'Train Acc: {logs["train_accs"][-1]:.3f}, '
                          f'Val Acc: {logs["val_accs"][-1]:.3f}')
                    
                    if logs['val_accs'][-1] > best_val_acc:
                        best_val_acc = logs['val_accs'][-1]
                        best_cores = [t.detach().clone()
                                      for t in tn_model.tensors]
                
                # Test
                tn_model = tk.models.MPSLayer(tensors=best_cores)
                tn_model.to(device)
                tn_model.eval()
                
                tn_model.trace(
                    torch.zeros(1, n_features - 1, embed_dim).to(device),
                    inline_input=False,
                    inline_mats=False
                )
                
                logs = test_tn(device=device,
                               model=tn_model,
                               embedding=embedding,
                               criterion=criterion,
                               test_loader=unused_loader,
                               logs=logs,
                               # n_batches=10,
                               )
                
                test_acc = logs['val_accs'][-1]
                
                logs = test_tn(device=device,
                               model=tn_model,
                               embedding=embedding,
                               criterion=criterion,
                               test_loader=sketch_loader,
                               logs=logs,
                               # n_batches=10,
                               )
                
                sketch_acc = logs['val_accs'][-1]
                
                print(f'**{model_class.name}** (p: {p_english}, i: {idx}, '
                      f's: {s}) => '
                      f'Test Acc.: {test_acc:.4f}, '
                      f'Sketch Acc.: {sketch_acc:.4f}')
                
                torch.save(
                    tn_model.tensors,
                    os.path.join(recores_dir,
                                 f'{s}_{test_acc:.4f}_{sketch_acc:.4f}.pt')
                )
                
                # Randomize parameters
                tn_model = tk.models.MPSLayer(tensors=best_cores)
                tn_model.to(device)
                tn_model.eval()
                
                # tn_model.canonicalize(mode='svdr', renormalize=True)
                
                for i, node in enumerate(tn_model.mats_env):
                    U = random_unitary(node.size('right')).to(device)
                    
                    if i < (len(tn_model.mats_env) - 1):
                        node.tensor = torch.einsum('lir,rk->lik', node.tensor, U)
                    
                    if i > 0:
                        node.tensor = torch.einsum('kl,lir->kir', prev_U, node.tensor)
                    
                    prev_U = U.clone().H
                
                tn_model.trace(
                    torch.zeros(1, n_features - 1, embed_dim).to(device),
                    inline_input=False,
                    inline_mats=False
                )
                
                logs = test_tn(device=device,
                               model=tn_model,
                               embedding=embedding,
                               criterion=criterion,
                               test_loader=unused_loader,
                               logs=logs,
                               # n_batches=10,
                               )
                
                test_acc = logs['val_accs'][-1]
                
                logs = test_tn(device=device,
                               model=tn_model,
                               embedding=embedding,
                               criterion=criterion,
                               test_loader=sketch_loader,
                               logs=logs,
                               # n_batches=10,
                               )
                
                sketch_acc = logs['val_accs'][-1]
                
                print(f'**{model_class.name}** (p: {p_english}, i: {idx}, '
                      f's: {s}) => '
                      f'Test Acc.: {test_acc:.4f}, '
                      f'Sketch Acc.: {sketch_acc:.4f}')
                
                torch.save(
                    tn_model.tensors,
                    os.path.join(priv_recores_dir,
                                 f'{s}_{test_acc:.4f}_{sketch_acc:.4f}.pt')
                )


###############################################################################
###############################################################################

###############
# System args #
###############
# MARK: System args

if __name__ == '__main__':
    argv = sys.argv
    if len(argv) == 1:
        print('No argumets were passed')
        print('Available options are:\n'
              '\t--help, -h\n'
              '\t<model name>\n'
              f'\t(optional) proportion of imbalance (p in {p_english_list})\n'
              '\t(optional) index of dataset (0 <= i <= 9)')
        sys.exit()
      
    # Read options and arguments
    try:
        opts, args = getopt.getopt(argv[1:], 'h', ['help'])
    except getopt.GetoptError:
        print('Available options are:\n'
              '\t--help, -h\n')
        sys.exit(2)

    model_name = None
    n_epochs = None
    p_english = None
    start_idx = None
    end_idx = None
    if len(args) == 2:
        model_name = args[0]
        n_epochs = int(args[1])
    elif len(args) == 3:
        model_name = args[0]
        n_epochs = int(args[1])
        if args[2].isdigit():
            start_idx = int(args[2])
            end_idx = int(args[2]) + 1
        else:
            p_english = float(args[2])
    elif len(args) == 4:
        model_name = args[0]
        n_epochs = int(args[1])
        p_english = float(args[2])
        start_idx = int(args[3])
        end_idx = int(args[3]) + 1
    elif len(args) == 5:
        model_name = args[0]
        n_epochs = int(args[1])
        p_english = float(args[2])
        start_idx = int(args[3])
        end_idx = int(args[4])
        
    aux_mod = import_file('model', os.path.join(cwd, 'models', f'{model_name}.py'))
    model_class = aux_mod.Model
    tensorize_retrain(model_class=model_class,
                      n_epochs=n_epochs,
                      p_english=p_english,
                      start_idx=start_idx,
                      end_idx=end_idx)
