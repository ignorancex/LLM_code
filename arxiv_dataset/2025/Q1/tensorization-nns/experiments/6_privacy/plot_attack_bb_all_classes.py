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
                #  sex,
                #  english,
                 transform=None):
        
        global cwd
        self.dataset = torchaudio.datasets.COMMONVOICE(
            root=os.path.join(cwd, 'CommonVoice'),
            tsv=os.path.join('datasets', 'full_df_test_copy.tsv'))
        self.transform = transform

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        x, y, z = self.dataset[index]
        if self.transform:
            x = self.transform((x, y))
        return x, (int(z['sex']), int(z['canadian']))
    

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


def load_data(batch_size):
    """Loads dataset performing the required transformations for train or test."""
    
    # Load datasets
    global transform
    test_dataset = CustomCommonVoice(transform=transform)
    
    # Create DataLoaders
    test_loader = DataLoader(test_dataset,
                             batch_size=batch_size,
                             collate_fn=none_collate,
                             shuffle=True)
    
    return test_loader


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

###################
# Create datasets #
###################
# MARK: Create datasets

n_samples = 75

def create_dataset():
    # Load data
    batch_size = 1000
    test_loader = load_data(batch_size)
    
    global n_samples
    all_n_samples = [n_samples] * 4
    all_ready = [False] * 4
    all_samples = [[], [], [], []]
    all_labels = [[], [], [], []]
    combs = [(0, 0), (0, 1), (1, 0), (1, 1)]
    
    for samples, (labels, p_targets) in test_loader:
        print(samples.size(0))
        for i, (l, p) in enumerate(combs):
            aux_idx = (labels == l) * (p_targets == p)
            aux_samples = samples[aux_idx][:all_n_samples[i]]
            aux_labels = labels[aux_idx][:all_n_samples[i]]
            
            all_samples[i].append(aux_samples)
            all_labels[i].append(aux_labels)
            
            all_n_samples[i] -= aux_samples.size(0)
            all_ready[i] = (all_n_samples[i] == 0)
            print(all_n_samples)
            
            if all_n_samples[i] < 0:
                print(f'Something went wrong! all_n_samples[{i}] < 0')
        
        if all(all_ready):
            break
        
    if not all(all_ready):
        raise ValueError('Not enough samples of all types')
        
    for i in range(4):
        all_samples[i] = torch.cat(all_samples[i], dim=0)
        all_labels[i] = torch.cat(all_labels[i], dim=0)
        print(all_samples[i].shape)
        
    all_samples = torch.cat(all_samples, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    print(all_samples.shape)
    
    aux_dir = os.path.join(cwd, 'results', '6_privacy', 'fffc_tiny',
                           'attacks', 'acc_by_class')
    os.makedirs(aux_dir, exist_ok=True)
    
    torch.save((all_samples, all_labels),
               os.path.join(aux_dir, 'test_samples.pt'))


def accs_by_class(model_type='nn',
                  retrained=False,
                  private=False):
    
    print(retrained, private)
    
    device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')
    
    aux_mod = import_file('model', os.path.join(cwd, 'models', 'fffc_tiny.py'))
    model_class = aux_mod.Model
    
    # Check tuned config of balanced model
    config_dir = os.path.join(cwd, 'results', '0_train_nns',
                              model_class.name, '0.5')
    with open(os.path.join(config_dir, 'tuned_config.json'), 'r') as f:
        config = json.load(f)
    
    all_results_dir = os.path.join(cwd, 'results', '6_privacy',
                                   'fffc_tiny', 'attacks', 'acc_by_class')
    os.makedirs(all_results_dir, exist_ok=True)
    
    softmax = nn.Softmax(dim=1)
    
    if model_type == 'nn':
        models_dir = os.path.join(cwd, 'results', '0_train_nns', 'fffc_tiny')
        results_dir = os.path.join(all_results_dir, 'nn')
        os.makedirs(results_dir, exist_ok=True)
    
    elif model_type == 'mps':
        
        def embedding(x):
            x = tk.embeddings.poly(x, degree=1)
            return x
        
        if not retrained and not private:
            models_dir = os.path.join(cwd, 'results', '6_privacy',
                                      'fffc_tiny', 'cores')
            results_dir = os.path.join(all_results_dir, 'cores')
            os.makedirs(results_dir, exist_ok=True)
        elif retrained and not private:
            models_dir = os.path.join(cwd, 'results', '6_privacy',
                                      'fffc_tiny', 'recores')
            results_dir = os.path.join(all_results_dir, 'recores')
            os.makedirs(results_dir, exist_ok=True)
        elif retrained and private:
            models_dir = os.path.join(cwd, 'results', '6_privacy',
                                      'fffc_tiny', 'priv_recores')
            results_dir = os.path.join(all_results_dir, 'priv_recores')
            os.makedirs(results_dir, exist_ok=True)
    
    # Load attack samples
    samples_dir = os.path.join(all_results_dir, 'test_samples.pt')
    if not os.path.exists(samples_dir):
        create_dataset()
    
    samples, labels = torch.load(samples_dir, weights_only=False)
    samples, labels = samples.to(device), labels.to(device)
    combs = [(0, 0), (0, 1), (1, 0), (1, 1)]
    
    for p_english in p_english_list:
        print(f'{p_english=}')
        p_accs = {}
        for idx in range(10):
            print(f'\t{idx=}')
            aux_models_dir = os.path.join(models_dir, str(p_english), str(idx))
            if os.path.exists(aux_models_dir):
                n_models = len(os.listdir(aux_models_dir))
                for s in range(n_models):
                    print(f'\t\t{s=}')
                    model_dir = list(filter(lambda f: f.startswith(f'{s}_'),
                                            os.listdir(aux_models_dir)))
                    
                    if model_dir:
                    
                        if model_type == 'nn':
                            state_dict = torch.load(os.path.join(aux_models_dir,
                                                                 model_dir[0]),
                                                    weights_only=False)
                            
                            model = model_class(config)
                            model.load_state_dict(state_dict)
                            model.eval()
                            model.to(device)
                            
                            results = softmax(model(samples))
                        
                        elif model_type == 'mps':
                            cores = torch.load(os.path.join(aux_models_dir,
                                                            model_dir[0]),
                                               weights_only=False)
                            
                            model = tk.models.MPSLayer(tensors=cores)
                            model.to(device)
                            model.eval()

                            model.trace(torch.zeros(1, out_rate // 2, 2).to(device),
                                        inline_input=False,
                                        inline_mats=False)
                            
                            results = model(embedding(samples),
                                            inline_input=False,
                                            inline_mats=False)
                            results = results.pow(2)
                            results = results / results.norm(dim=1, keepdim=True)
                        
                        _, preds = torch.max(results, 1)
                        accuracy = (preds == labels).float().mean().item()
                        
                        for l, p in combs:
                            aux_idx = 2*l + p
                            aux_res = results[(aux_idx * n_samples):((aux_idx + 1) * n_samples)]
                            aux_lab = labels[(aux_idx * n_samples):((aux_idx + 1) * n_samples)]
                            
                            _, preds = torch.max(aux_res, 1)
                            accuracy = (preds == aux_lab).float().mean().item()
                            
                            if (l, p) in p_accs:
                                p_accs[(l, p)].append(accuracy)
                            else:
                                p_accs[(l, p)] = [accuracy]
        
        for l, p in combs:
            accs = p_accs[(l, p)]
        
            accs = torch.Tensor(accs)
            accs = (accs.mean(), accs.std())
            print(f'\t\t\t{accs}')
            
            aux_dir = os.path.join(results_dir, str(p_english))
            os.makedirs(aux_dir, exist_ok=True)
            
            torch.save(accs, os.path.join(aux_dir, f'{l}_{p}.pt'))


###############
# System args #
###############
# MARK: System args

if __name__ == '__main__':
    argv = sys.argv
    if len(argv) == 1:
        print('No argumets were passed')
        print('\t<sex>\n'
              '\t<english>')
        sys.exit()
      
    # Read options and arguments
    try:
        opts, args = getopt.getopt(argv[1:], 'h', ['help'])
    except getopt.GetoptError:
        print('Available options are:\n'
              '\t--help, -h\n')
        sys.exit(2)
    
    model_type = None
    retrained = None
    private = None
    if len(args) == 3:
        model_type = args[0]
        retrained = bool(int(args[1]))
        private = bool(int(args[2]))
    else:
        print('All arguments should be passed')
        print('\t<model_type>\n'
              '\t<retrained>\n'
              '\t<private>')
        sys.exit()
        
    accs_by_class(model_type=model_type,
                  retrained=retrained,
                  private=private)
