"""
Tensorization of NN models for different configurations of hyperparameters
"""

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


def test_tn(device, model, embedding, test_loader, n_batches=None):
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
            
            scores = model(embedding(data),
                           inline_input=False,
                           inline_mats=False)
            scores = scores.pow(2)
            scores = scores / scores.norm(dim=1, keepdim=True)
            
            _, preds = torch.max(scores, 1)
            
            accuracy = (preds == labels).float().mean().item()
            running_acc += accuracy
            
            i += 1
            if i >= n_batches:
                break
    
    return running_acc / n_batches


###############################################################################
###############################################################################

######################
# Mutual Information #
######################
# MARK: Mutual Information

@torch.no_grad()
def estimate_mi(fun, cut, samples, device):
    """
    Estimates MI of function fun using only points in dataset.

    Parameters
    ----------
    fun : function
        Function returning values in [0, 1].
    cut : int
        Indices < cut constitute the left subsystem, indices >= cut
        constitute the right subsystem. MI is computed for the correlation
        between left and right subsystems.
    samples : torch.Tensor
        Samples used to estimate MI. They include labels. It has shape
        ``batch x (in_dim + 1)``
    """
    samples = samples.to(device)
        
    # Obtain unique left and right parts of samples in dataset
    left = samples[:, :cut].unique(dim=0)  # batch x dim
    right = samples[:, cut:].unique(dim=0) # batch x dim
    
    # Create mesh grid where each row shares the same left part,
    # and each column shares the same right part in the samples
    left_aux = left.unsqueeze(1).expand(-1, right.size(0), -1)
    right_aux = right.unsqueeze(0).expand(left.size(0), -1, -1)

    mesh = torch.cat([left_aux, right_aux], dim=2)
    
    # Evaluate fun on all points of the grid
    values = fun(mesh.view(left.size(0) * right.size(0), -1))
    values = values.view(mesh.size(0), mesh.size(1))
    values = values / values.sum()  # Normalize to have distribution
    
    # Sum values in each row (share left part) and each column (share right part)
    partials_left = values.sum(1)
    partials_right = values.sum(0)
    
    # Outer product of partials_left and partials_right
    cross_values = torch.outer(partials_left, partials_right)
    
    # Compute MI
    logs = torch.log(torch.div(values, cross_values))
    logs = torch.where(torch.logical_or(logs.isnan(), logs.isinf()), 0., logs)
    mi = torch.sum(torch.mul(values, logs))
    
    return mi


def estimate_mi_all_cuts_aux(fun, samples, device):
    lst_mis = []
    # Dim of vectors in dataset is equal to number of sites in MPS
    for cut in range(1, samples.size(1)): 
        lst_mis.append(estimate_mi(fun, cut, samples, device))
    return torch.stack(lst_mis, dim=0)


def estimate_mi_all_cuts(fun, samples, device):
    mi_tensor = estimate_mi_all_cuts_aux(fun, samples, device)
    return mi_tensor.mean().item(), mi_tensor.std().item(), mi_tensor.max().item()


###############################################################################
###############################################################################

#############
# Tensorize #
#############
# MARK: Tensorize

def tensorize(model_class,
              embedding_fn='poly',
              embed_dim=2,
              bond_dim=5,
              domain_multiplier=3,
              n_samples=100,
              n_models=10):
    """Tensorizes models with given parameters."""
    device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')
    
    # Check tuned config of balanced model
    config_dir = os.path.join(cwd, 'results', '0_train_nns',
                              'fffc_tiny', '0.5')
    with open(os.path.join(config_dir, 'tuned_config.json'), 'r') as f:
        config = json.load(f)
    
    softmax = nn.Softmax(dim=1)
        
    models_dir = os.path.join(cwd, 'results', '3_hyperparameters')
    cores_dir = os.path.join(models_dir,
                             f'cores_{model_class.name}',
                             f'{embed_dim}_{bond_dim}_{domain_multiplier}_{n_samples}')
    os.makedirs(cores_dir, exist_ok=True)
    
    n_cores = len(os.listdir(cores_dir))
    
    for s in range(n_cores, n_models):
        # Load state_dict
        state_dict_dir = list(
            filter(lambda f: f.startswith(model_class.name) and f.endswith('.pt'),
                   os.listdir(models_dir)))[0]
        state_dict = torch.load(os.path.join(models_dir, state_dict_dir),
                                weights_only=False)
        
        # Initialize model with balanced config
        model = model_class(config)
        model.load_state_dict(state_dict)
        model.eval()
        model.to(device)
        
        # Function to tensorize
        def fn(samples):
            aux = softmax(model(samples))
            aux = torch.where(aux > 0.5, 1., 0.)
            return aux
                
            # return softmax(model(samples)).sqrt()
        
        # Load data
        batch_size_loader = 500
        tensorize_loader, unused_loader = load_sketch_samples(
            0.5, 0, batch_size_loader)
        sketch_samples, sketch_labels = next(iter(tensorize_loader))
        perm = torch.randperm(sketch_samples.size(0))
        sketch_samples = sketch_samples[perm]
        sketch_labels = sketch_labels[perm]
        
        n_features = sketch_samples.shape[1] + 1
        
        # Tensorization hyperprameters
        cum_percentage = 0.99
        out_position = n_features // 2
        
        domain_dim = domain_multiplier * embed_dim
        domain = torch.arange(domain_dim).float() / domain_dim
        
        batch_size_tensorize = 1000

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
        
        # Tensorize
        cores = tt_rss(function=fn,
                       embedding=embedding,
                       sketch_samples=sketch_samples[:n_samples],
                       labels=sketch_labels[:n_samples],
                       domain=domain,
                       domain_multiplier=domain_multiplier,
                       out_position=out_position,
                       rank=bond_dim,
                       cum_percentage=cum_percentage,
                       batch_size=batch_size_tensorize,
                       device=device,
                       verbose=False)
        
        tn_model = tk.models.MPSLayer(tensors=cores)
        tn_model.to(device)
        tn_model.eval()
        
        tn_model.trace(
            torch.zeros(1, n_features - 1, embed_dim).to(device),
            inline_input=False,
            inline_mats=False
        )
        
        # Check accuracy of TN
        # Train
        sketch_dataset = TensorDataset(sketch_samples[:n_samples],
                                       sketch_labels[:n_samples])
        sketch_loader = DataLoader(sketch_dataset,
                                   batch_size=n_samples,
                                   collate_fn=none_collate,
                                   shuffle=False)
        
        sketch_acc = test_tn(device=device,
                             model=tn_model,
                             embedding=embedding,
                             test_loader=sketch_loader,
                             #n_batches=5
                             )
        
        # Test
        test_acc = test_tn(device=device,
                           model=tn_model,
                           embedding=embedding,
                           test_loader=unused_loader,
                           #n_batches=5
                           )
        
        # Check MI of NN
        def mi_fn(samples):
            labels = samples[:, out_position:(out_position + 1)].to(torch.int64)
            samples = torch.cat([samples[:, :out_position],
                                 samples[:, (out_position + 1):]],
                                dim=1)
            result = softmax(model(samples))
            outputs = result.gather(dim=1, index=labels).flatten()
            return outputs
        
        mi_samples = torch.cat([sketch_samples[:, :out_position],
                                sketch_labels.unsqueeze(1).float(),
                                sketch_samples[:, out_position:]],
                               dim=1)[:n_samples]
        
        mean, std, max = estimate_mi_all_cuts(mi_fn, mi_samples, device)
        max_bond_dim = torch.tensor(max).exp()
        
        # Print logs
        print(f'**{model_class.name}** (p: 0.5, i: 0, s: {s}) => '
              f'Sketch Acc.: {sketch_acc:.4f}, '
              f'Test Acc.: {test_acc:.4f}, '
              f'MI: {mean=:.4f} / {std=:.4f} / {max=:.4f} / '
              f'{max_bond_dim=:.4}'
             )
        
        torch.save(
            cores,
            os.path.join(
                cores_dir,
                f'{s}_{sketch_acc:.4f}_{test_acc:.4f}_{mean:.4f}_{std:.4f}_{max:.4f}.pt'
            )
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
              '\t<embedding name>\n'
              '\t<embed dim>\n'
              '\t<bond dim>\n'
              '\t<domain multiplier>\n'
              '\t<n samples>\n'
              '\t<n models>\n')
        sys.exit()
      
    # Read options and arguments
    try:
        opts, args = getopt.getopt(argv[1:], 'h', ['help'])
    except getopt.GetoptError:
        print('Available options are:\n'
              '\t--help, -h\n')
        sys.exit(2)

    model_name = None
    embedding_fn = None
    embed_dim = None
    bond_dim = None
    domain_multiplier = None
    n_samples = None
    n_models = None
    if len(args) == 1:
        model_name = args[0]
    elif len(args) == 7:
        model_name = args[0]
        embedding_fn = args[1]
        if embedding_fn not in ['poly', 'unit', 'basis', 'fourier']:
            print('Available options for <embedding name> are:')
            print('\tpoly\n'
                  '\tunit\n'
                  '\tbasis\n')
        embed_dim = int(args[2])
        bond_dim = int(args[3])
        domain_multiplier = int(args[4])
        n_samples = int(args[5])
        n_models = int(args[6])
    else:
        print('All arguments should be passed')
        print('\t<model name>\n'
              '\t<embedding name>\n'
              '\t<embed dim>\n'
              '\t<bond dim>\n'
              '\t<domain multiplier>\n'
              '\t<n samples>\n'
              '\t<n models>\n')
        sys.exit()
        
    aux_mod = import_file('model', os.path.join(cwd, 'models', f'{model_name}.py'))
    model_class = aux_mod.Model
    tensorize(model_class=model_class,
              embedding_fn=embedding_fn,
              embed_dim=embed_dim,
              bond_dim=bond_dim,
              domain_multiplier=domain_multiplier,
              n_samples=n_samples,
              n_models=n_models)
