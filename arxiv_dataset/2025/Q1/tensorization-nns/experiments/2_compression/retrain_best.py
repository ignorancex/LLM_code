"""
Re-train tensorized MPS models
"""

import os
import sys
import getopt
from importlib import util

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torchvision
import torchaudio

import tensorkrowch as tk

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
                                       batch_size=batch_size,
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

def n_params(model):
    n = 0
    for p in model.parameters():
        n += p.numel()
    return n


def retrain_tn(model_class, bond_dim, n_epochs):
    """Retrains best MPS models"""
    global p_english_list
    
    device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')
    
    cores_dir = os.path.join(cwd, 'results', '2_compression',
                             f'cores_{model_class.name}',
                             f'2_{bond_dim}_1_100')
    all_files = os.listdir(cores_dir)
    all_accs = [float(f.split('_')[1])for f in all_files]
    
    aux_cores_dir = list(
        filter(lambda f: float(f.split('_')[1]) == max(all_accs),
               all_files))[0]
    
    cores = torch.load(os.path.join(cores_dir, aux_cores_dir),
                       weights_only=False)
    prev_test_acc = float(aux_cores_dir.split('_')[1])
    
    recores_dir = os.path.join(cwd, 'results', '2_compression',
                               f'recores_{model_class.name}')
    os.makedirs(recores_dir, exist_ok=True)
                
    # Load data
    batch_size = 64
    train_loader, val_loader, test_loader = load_data(0.5, 0, batch_size)
    tensorize_loader, unused_loader = load_sketch_samples(0.5, 0, batch_size)
    
    n_features = out_rate // 2 + 1
    embed_dim = 2
    
    def embedding(x):
        x = tk.embeddings.poly(x, degree=embed_dim - 1)
        return x
    
    # MPS model
    tn_model = tk.models.MPSLayer(tensors=cores)
    tn_model.to(device)
    
    tn_model.trace(
        torch.zeros(1, n_features - 1, embed_dim).to(device),
        inline_input=False,
        inline_mats=False
    )
    
    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(tn_model.parameters(),
                                 lr=1e-5,
                                 weight_decay=1e-8)
    
    logs = {'train_losses': [],
            'val_losses': [],
            'train_accs': [],
            'val_accs': []}
    best_val_acc = -1.
    
    print(f'**{model_class.name}** (p: {0.5}, i: {0}, D: {bond_dim}) => '
          f'Prev Test Acc.: {prev_test_acc:.4f}')
    
    for epoch in range(n_epochs):
        tn_model, logs = training_epoch_tn(
            device=device,
            model=tn_model,
            embedding=embedding,
            criterion=criterion,
            optimizer=optimizer,
            train_loader=train_loader, #tensorize_loader,
            logs=logs,
            # n_batches=10,
            )
        
        # Validate
        logs = test_tn(device=device,
                       model=tn_model,
                       embedding=embedding,
                       criterion=criterion,
                       test_loader=val_loader, #unused_loader,
                       logs=logs,
                       n_batches=5,
                       )
        
        print(f'**{model_class.name}** (p: {0.5}, i: {0}, D: {bond_dim}) => '
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
    
    print(f'**{model_class.name}** (p: {0.5}, i: {0}, D: {bond_dim}) => '
          f'Test Acc.: {test_acc:.4f}')
    
    if test_acc < prev_test_acc:
        print('** Not Improved **')
        
        best_cores = cores
        tn_model = tk.models.MPSLayer(tensors=cores)
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
    
        print(f'**{model_class.name}** (p: {0.5}, i: {0}, D: {bond_dim}) => '
              f'Test Acc.: {test_acc:.4f}')
    
    n = n_params(tn_model)
    
    torch.save(
        tn_model.tensors,
        os.path.join(
            recores_dir,
            f'{bond_dim}_{n_epochs}_{n}_{prev_test_acc:.4f}_{test_acc:.4f}.pt')
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
              '\t<bond dim>\n'
              '\t<n epochs>')
        sys.exit()
      
    # Read options and arguments
    try:
        opts, args = getopt.getopt(argv[1:], 'h', ['help'])
    except getopt.GetoptError:
        print('Available options are:\n'
              '\t--help, -h\n')
        sys.exit(2)

    model_name = None
    bond_dim = None
    n_epochs = None
    if len(args) == 3:
        model_name = args[0]
        bond_dim = int(args[1])
        n_epochs = int(args[2])
    else:
        print('All arguments have to be passed')
        print('Available options are:\n'
              '\t--help, -h\n'
              '\t<model name>\n'
              '\t<bond dim>\n'
              '\t<n epochs>')
        sys.exit()
        
        
    aux_mod = import_file('model', os.path.join(cwd, 'models', f'{model_name}.py'))
    model_class = aux_mod.Model
    retrain_tn(model_class=model_class,
               bond_dim=bond_dim,
               n_epochs=n_epochs)
