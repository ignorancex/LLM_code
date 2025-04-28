"""
Train MPS from tensorized pre-trained fffc_tiny model
"""

import os
import sys
import getopt
import json
from importlib import util

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchaudio

import tensorkrowch as tk
from tensorkrowch.decompositions import tt_rss

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


def training_epoch(device, model, criterion, optimizer, train_loader,
                   n_batches=None):
    if n_batches is not None:
        n_batches = min(n_batches, len(train_loader))
    else:
        n_batches = len(train_loader)
    i = 0
    
    model.train()
    for data, labels in train_loader:
        data = data.to(device)
        labels = labels.to(device)
        
        scores = model(data)
        loss = criterion(scores, labels)
        
        _, preds = torch.max(scores, 1)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient descent
        optimizer.step()
        
        i += 1
        if i >= n_batches:
            break
    
    return model


def training_epoch_tn(device, model, embedding, renormalize,
                      criterion, optimizer, train_loader, logs, n_batches=None):
    print_each = len(train_loader) // 10
    
    if n_batches is not None:
        n_batches = min(n_batches, len(train_loader))
    else:
        n_batches = len(train_loader)
    i = 0

    model.train()
    for batch_idx, (data, labels) in enumerate(train_loader):
        data = data.to(device)
        labels = labels.to(device)
        
        # Forward
        scores = model(embedding(data),
                       inline_input=False,
                       inline_mats=False,
                       renormalize=renormalize)
        scores = scores.pow(2)
        scores = scores / scores.norm(dim=1, keepdim=True)
        scores = torch.where(scores == 0, 1e-10, scores)
        scores = scores.log()
        
        loss = criterion(scores, labels)
        
        with torch.no_grad():
            _, preds = torch.max(scores, 1)
            accuracy = (preds == labels).float().mean().item()
            
            logs['train_losses'].append(loss.item())
            logs['train_accs'].append(accuracy)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient descent
        optimizer.step()
        
        if ((batch_idx + 1) % print_each == 0):
            print(f'\tBatch: {batch_idx + 1}/{len(train_loader)}, '
                  f'Last Train Loss: {loss.item():.3f}, '
                  f'Last Train Acc: {accuracy:.3f}')
        
        i += 1
        if i >= n_batches:
            break
    
    return model, logs


def test_tn(device, model, embedding, renormalize, criterion, test_loader,
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
                           inline_mats=False,
                           renormalize=renormalize)
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

def train_tn(init_method='rss',
             embedding_fn='poly',
             renormalize=False,
             bond_dim=5,
             n_epochs=5,
             learning_rate=1e-3,
             weight_decay=1e-6):
    """Retrains best MPS models"""
    print(init_method, embedding_fn, renormalize, bond_dim,
          n_epochs, learning_rate, weight_decay)
    
    global p_english_list
    
    device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')
    
    # Tensorization hyperprameters
    n_features = out_rate // 2 + 1
    embed_dim = 2
    
    if embedding_fn == 'poly':
        def embedding(x):
            x = tk.embeddings.poly(x, degree=embed_dim - 1)
            return x
        
    elif embedding_fn == 'unit':
        def embedding(x):
            x = tk.embeddings.unit(x, dim=embed_dim)
            return x
    
    elif embedding_fn == 'basis':
        def embedding(x):
            x = tk.embeddings.discretize(x, base=embed_dim, level=1).squeeze(-1).int()
            x = tk.embeddings.basis(x, dim=embed_dim).float() # batch x n_features x dim
            return x
    
    elif embedding_fn == 'fourier':
        def embedding(x):
            x = tk.embeddings.fourier(x, dim=embed_dim)
            return x
    
    if init_method.startswith('rss'):
        
        softmax = nn.Softmax(dim=1)
        
        if init_method == 'rss':
            # Initialize from tensorization of fffc_tiny model
            aux_mod = import_file('model', os.path.join(cwd, 'models', 'fffc_tiny.py'))
            model_class = aux_mod.Model
            
            models_dir = os.path.join(cwd, 'results', '0_train_nns',
                                    model_class.name, '0.5', '0')
            
            # Check tuned config of balanced model
            config_dir = os.path.join(cwd, 'results', '0_train_nns',
                                    model_class.name, '0.5')
            with open(os.path.join(config_dir, 'tuned_config.json'), 'r') as f:
                config = json.load(f)
            
            state_dict_dir = os.listdir(models_dir)[0]
            state_dict = torch.load(os.path.join(models_dir, state_dict_dir),
                                    weights_only=False)
            
            # Initialize model with balanced config
            model = model_class(config)
            model.load_state_dict(state_dict)
            model.eval()
            model.to(device)
            
            def fn(samples):
                return softmax(model(samples)).sqrt()
            
            n_samples = 50
        
        elif init_method == 'rss_random':
            
            def fn(samples):
                result = torch.randn(samples.size(0), 2).to(device)
                return softmax(result).sqrt()
            
            n_samples = 10
        
        elif init_method == 'rss_pretrain':
            # Initialize from tensorization of fffc_tiny model
            aux_mod = import_file('model', os.path.join(cwd, 'models', 'fffc_micro.py'))
            model_class = aux_mod.Model
            
            # Check tuned config of balanced model
            config_dir = os.path.join(cwd, 'results', '0_train_nns',
                                      'fffc_tiny', '0.5')
            with open(os.path.join(config_dir, 'tuned_config.json'), 'r') as f:
                config = json.load(f)
            
            # Initialize model with balanced config
            model = model_class(config)
            model.to(device)
            
            # Load data
            batch_size = 32
            train_loader, val_loader, _ = load_data(0.5, 0, batch_size)
            
            # Pre-train model for a few epochs
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(),
                                         lr=1e-1,
                                         weight_decay=1e-3)
            
            model = training_epoch(device=device,
                                   model=model,
                                   criterion=criterion,
                                   optimizer=optimizer,
                                   train_loader=train_loader,
                                   n_batches=10,
                                   )
            
            print('* Finished pre-training')
            
            def fn(samples):
                return softmax(model(samples)).sqrt()
            
            n_samples = 10
        
        # Load data
        batch_size_loader = 500
        tensorize_loader, unused_loader = load_sketch_samples(
            0.5, 0, batch_size_loader)
        sketch_samples, sketch_labels = next(iter(tensorize_loader))
        perm = torch.randperm(sketch_samples.size(0))
        sketch_samples = sketch_samples[perm]
        sketch_labels = sketch_labels[perm]
        
        # Tensorization hyperprameters
        cum_percentage = 0.99
        domain_multiplier = 2
        
        domain_dim = domain_multiplier * embed_dim
        domain = torch.arange(domain_dim).float() / domain_dim
        
        batch_size_tensorize = 1000
        
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
        
        print('* Finished tensorization')
        
        # MPS model
        tn_model = tk.models.MPSLayer(tensors=cores)
        tn_model.to(device)
        
        tn_model.canonicalize(renormalize=True)
        
        tn_model.trace(
            torch.zeros(1, n_features - 1, embed_dim).to(device),
            inline_input=False,
            inline_mats=False,
            renormalize=renormalize
        )
    
    else:
        # MPS model
        tn_model = tk.models.MPSLayer(n_features=n_features,
                                      in_dim=embed_dim,
                                      out_dim=2,
                                      bond_dim=bond_dim,
                                      init_method=init_method,
                                      std=1e-5)
        tn_model.to(device)
        
        tn_model.trace(
            torch.zeros(1, n_features - 1, embed_dim).to(device),
            inline_input=False,
            inline_mats=False,
            renormalize=renormalize
        )
    
    # Load data
    batch_size = 32
    train_loader, val_loader, _ = load_data(0.5, 0, batch_size)
    
    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(tn_model.parameters(),
                                 lr=learning_rate,
                                 weight_decay=weight_decay)
    
    logs = {'train_losses': [],
            'val_losses': [],
            'train_accs': [],
            'val_accs': []}
    
    for epoch in range(n_epochs):
        tn_model, logs = training_epoch_tn(
            device=device,
            model=tn_model,
            embedding=embedding,
            renormalize=renormalize,
            criterion=criterion,
            optimizer=optimizer,
            train_loader=train_loader,
            logs=logs,
            # n_batches=5,
            )
        
        logs = test_tn(device=device,
                       model=tn_model,
                       embedding=embedding,
                       renormalize=renormalize,
                       criterion=criterion,
                       test_loader=val_loader,
                       logs=logs,
                       # n_batches=5
                       )
        
        print(f'**Epoch: {epoch + 1}/{n_epochs}** => '
              f'Train Loss: {logs["train_losses"][-1]:.3f}, '
              f'Val Loss: {logs["val_losses"][-1]:.3f}, '
              f'Train Acc: {logs["train_accs"][-1]:.3f}, '
              f'Val Acc: {logs["val_accs"][-1]:.3f}')
        print('\t', init_method, embedding_fn, renormalize, bond_dim)
    
    results_dir = os.path.join(cwd, 'results', '5_initialization')
    os.makedirs(results_dir, exist_ok=True)
    
    torch.save(
        (logs, tn_model.tensors),
        os.path.join(results_dir,
                     f'{init_method}_{embedding_fn}_{renormalize}_'
                     f'{bond_dim}_{n_epochs}_{learning_rate:.2e}_'
                     f'{weight_decay:.2e}.pt')
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
              '\t<init method>\n'
              '\t<embedding>\n'
              '\t<renormalize>\n'
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
    
    init_method = None
    embedding_fn = None
    renormalize = None
    bond_dim = None
    n_epochs = None
    learning_rate = None
    weight_decay = None
    if len(args) == 7:
        init_method = args[0]
        embedding_fn = args[1]
        renormalize = bool(int(args[2]))
        bond_dim = int(args[3])
        n_epochs = int(args[4])
        learning_rate = float(args[5])
        weight_decay = float(args[6])
    else:
        print('All arguments have to be passed')
        print('Available options are:\n'
              '\t--help, -h\n'
              '\t<init method>\n'
              '\t<embedding>\n'
              '\t<renormalize>\n'
              '\t<bond dim>\n'
              '\t<n epochs>')
        sys.exit()
    
    train_tn(init_method=init_method,
             embedding_fn=embedding_fn,
             renormalize=renormalize,
             bond_dim=bond_dim,
             n_epochs=n_epochs,
             learning_rate=learning_rate,
             weight_decay=weight_decay)
