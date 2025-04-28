"""
Layer-wise tensorization of NN models with different bond dimensions,
and posterior training of these tensorized models
"""

import os
import sys
import getopt
import copy
from importlib import util

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchaudio

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


###############################################################################
###############################################################################

############
# Training #
############
# MARK: Training

def training_epoch(device, model, criterion, optimizer, train_loader,
                   logs, verbose=False):
    running_loss = 0
    running_acc = 0
    print_each = len(train_loader) // 10
    
    model.train()
    for batch_idx, (data, labels) in enumerate(train_loader):
        data = data.to(device)
        labels = labels.to(device)
        
        # Forward
        scores = model(data)
        loss = criterion(scores, labels)
        
        with torch.no_grad():
            _, preds = torch.max(scores, 1)
            accuracy = (preds == labels).float().mean().item()
            running_loss += loss.item()
            running_acc += accuracy
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient descent
        optimizer.step()
        
        if verbose and ((batch_idx + 1) % print_each == 0):
            print(f'\tBatch: {batch_idx + 1}/{len(train_loader)}, '
                  f'Last Train Loss: {loss.item():.3f}, '
                  f'Last Train Acc: {accuracy:.3f}')
    
    logs['train_losses'].append(running_loss / len(train_loader))
    logs['train_accs'].append(running_acc / len(train_loader))
    
    return model, optimizer, logs


def test(device, model, test_loader, n_batches=None):
    """Computes accuracy on test set."""
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
            
            scores = model(data)
            _, preds = torch.max(scores, 1)
            
            accuracy = (preds == labels).float().mean().item()
            running_acc += accuracy
            
            i += 1
            if i >= n_batches:
                break
    
    return running_acc / n_batches


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


def test_and_retrain(model_class, bond_dim=2, n_epochs=5):
    """Tensorizes models with given parameters."""
    device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')
        
    models_dir = os.path.join(cwd, 'results', '2_compression')
    tn_models_dir = os.path.join(models_dir, f'{model_class.name}')
    os.makedirs(tn_models_dir, exist_ok=True)
    
    # Initialize model with balanced config
    model = model_class(bond_dim=bond_dim)
    model.to(device)
    
    # Load data
    batch_size = 32
    train_loader, val_loader, test_loader = load_data(0.5, 0, batch_size)
    tensorize_loader, unused_loader = load_sketch_samples(0.5, 0, batch_size)
        
    # Check accuracy of TN
    prev_test_acc = test(device=device,
                         model=model,
                         test_loader=unused_loader)
    n = n_params(model)
    
    # Print logs
    print(f'**{model_class.name}** (p: 0.5, i: 0) => '
          f'Test Acc.: {prev_test_acc:.4f}, No. Params.: {n}')
    
    # Re-train
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-6,
        )
    
    logs = {'train_losses': [],
            'val_losses': [],
            'train_accs': [],
            'val_accs': []}
    best_val_acc = -1.
    
    for epoch in range(n_epochs):
        model, optimizer, logs = training_epoch(
            device=device,
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            train_loader=train_loader, #tensorize_loader,
            logs=logs)
    
        val_acc = test(device=device,
                       model=model,
                       test_loader=val_loader,
                       n_batches=5)

        print(f'**{model_class.name}** (p: 0.5, i: 0) => '
              f'Epoch: {epoch + 1}/{n_epochs}, '
              f'Train Loss: {logs["train_losses"][-1]:.3f}, '
              f'Train Acc: {logs["train_accs"][-1]:.3f}, '
              f'Val Acc: {val_acc:.3f}')

        # Keep track of best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state_dict = copy.deepcopy(model.state_dict())
        
    model.load_state_dict(best_model_state_dict)
    test_acc = test(device=device,
                    model=model,
                    test_loader=unused_loader)
    
    # Print logs
    print(f'**{model_class.name}** (p: 0.5, i: 0) => Test Acc.: {test_acc:.4f}')
    
    if test_acc < prev_test_acc:
        print('** Not Improved **')
        
        model = model_class(bond_dim=bond_dim)
        model.to(device)
        
        test_acc = test(device=device,
                        model=model,
                        test_loader=unused_loader)
        
        # Print logs
        print(f'**{model_class.name}** (p: 0.5, i: 0) => Test Acc.: {test_acc:.4f}')
        
    # Save best model after training
    torch.save(
        model.state_dict(),
        os.path.join(
            tn_models_dir,
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
              '\t<n epochs>\n')
        sys.exit()
      
    # Read options and arguments
    try:
        opts, args = getopt.getopt(argv[1:], 'h', ['help'])
    except getopt.GetoptError:
        print('Available options are:\n' # TODO: I'm not doing anything with help
              '\t--help, -h\n')
        sys.exit(2)

    model_name = None
    bond_dim = None
    n_epochs = None
    if len(args) == 1:
        model_name = args[0]
    elif len(args) == 3:
        model_name = args[0]
        bond_dim = int(args[1])
        n_epochs = int(args[2])
    else:
        print('All arguments should be passed')
        print('\t<model name>\n'
              '\t<bond dim>\n'
              '\t<n epochs>\n')
        sys.exit()
        
    aux_mod = import_file('model', os.path.join(cwd, 'models', f'{model_name}.py'))
    model_class = aux_mod.Model
    test_and_retrain(model_class=model_class,
                     bond_dim=bond_dim,
                     n_epochs=n_epochs)
