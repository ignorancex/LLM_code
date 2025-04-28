import os
import sys
import getopt

import torch
import torch.nn as nn

import torchvision.datasets as datasets
import torchvision.transforms as transforms

import tensorkrowch as tk
from tensorkrowch.decompositions import tt_rss


torch.set_num_threads(1)
cwd = os.getcwd()
eps = 1e-10


# Train NN models
# ===============

class FFFC(nn.Module):
    
    def __init__(self, im_size):
        super().__init__()
        
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(im_size ** 2, 500)
        self.fc2 = nn.Linear(500, 200)
        self.fc3 = nn.Linear(200, 10)
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def check_accuracy(loader, model, device, im_size):
    num_correct = 0
    num_samples = 0
    
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            
            x = x.view(-1, im_size**2)
            
            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)
        
    model.train()
    return float(num_correct)/float(num_samples)*100


def train_model(n_features):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    cores_dir = os.path.join(cwd, 'results', '1_performance', 'mnist')
    os.makedirs(cores_dir, exist_ok=True)

    batch_size = 128
    im_size = n_features

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Resize(im_size, antialias=True),
                                    ])
    train_dataset = datasets.MNIST(root=os.path.join(cores_dir, 'data'),
                                   train=True,
                                   download=True,
                                   transform=transform)
    test_dataset = datasets.MNIST(root=os.path.join(cores_dir, 'data'),
                                  train=False,
                                  download=True,
                                  transform=transform)

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=batch_size,
                                              shuffle=True)
    
    # Initialize network
    model = FFFC(im_size).to(device)

    # Hyperparameters
    n_epochs = 10
    learning_rate = 1e-3
    weight_decay = 1e-5

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=learning_rate,
                                 weight_decay=weight_decay)
    # Train network
    print('* TRAINING MODEL...')
    for epoch in range(n_epochs):
        for batch_idx, (data, targets) in enumerate(train_loader):
            # Get data to cuda if possible
            data = data.to(device)
            targets = targets.to(device)
            
            data = data.view(-1, im_size**2)
            
            # Forward
            scores = model(data)
            loss = criterion(scores, targets)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient descent
            optimizer.step()
            
        # train_acc = check_accuracy(train_loader, model)
        test_acc = check_accuracy(test_loader, model, device, im_size)
            
        print(f'Epoch {epoch+1}/{n_epochs} => '
              f'Train Loss: {loss:.4f}, Test Acc.: {test_acc:.2f}')

    # Save model
    torch.save(model.state_dict(),
               os.path.join(cores_dir, f'fffc_mnist_{im_size}.pt'))


# Tensorize
# =========

def tt_rss_tensorization(n_features, phys_dim, bond_dim,
                         samples_size, sketch_size, verbose=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    cores_dir = os.path.join(cwd, 'results', '1_performance', 'mnist')
    os.makedirs(cores_dir, exist_ok=True)
    
    results_dir = os.path.join(
        cwd, 'results', '1_performance', 'mnist',
        f'rss_{n_features}_{phys_dim}_{bond_dim}_{samples_size}_{sketch_size}')
    os.makedirs(results_dir, exist_ok=True)
    
    cores_file = f'fffc_mnist_{n_features}.pt'
    if not os.path.exists(os.path.join(cores_dir, cores_file)):
        train_model(n_features=n_features)
    
    model_state_dict = torch.load(os.path.join(cores_dir, cores_file),
                                  weights_only=False)
    
    model = FFFC(n_features)
    model.load_state_dict(model_state_dict)
    model.to(device)
    
    transform = transforms.Resize(n_features, antialias=True)
    train_dataset = datasets.MNIST(root=os.path.join(cores_dir, 'data'),
                                   train=True,
                                   download=True)
    test_dataset = datasets.MNIST(root=os.path.join(cores_dir, 'data'),
                                  train=False,
                                  download=True)
    
    sketch_samples = transform(train_dataset.data).view(-1, n_features**2) / 255
    sketch_samples = sketch_samples[:sketch_size]
    
    samples = transform(test_dataset.data).view(-1, n_features**2) / 255
    samples = samples[:samples_size]
    
    softmax = nn.Softmax(dim=1)
    
    def fun(input):
        output = softmax(model(input)).sqrt()
        return output
    
    def embedding(x):
        x = tk.embeddings.poly(x, degree=phys_dim - 1)
        return x
    
    domain = torch.arange(phys_dim).float() / phys_dim
    
    if verbose: print('* Starting TT-RSS tensorization...')
    cores_rss, info = tt_rss(function=fun,
                             embedding=embedding,
                             sketch_samples=sketch_samples,
                             domain=domain,
                             rank=bond_dim,
                             cum_percentage=1-1e-5,
                             batch_size=500,
                             device=device,
                             verbose=verbose,
                             return_info=True)
    
    if verbose:
        print(f'* Tensorization finished in {info["total_time"]:.2e} seconds')
        print(f'--> Sketch error: {info["val_eps"]:.2e}')
    
    # Relative error
    mps_rss = tk.models.MPSLayer(tensors=[c.to(device) for c in cores_rss])
    mps_rss.trace(torch.zeros(1, n_features**2, phys_dim, device=device))
    
    exact_results = fun(samples.to(device))
    rss_results = mps_rss(embedding(samples.to(device)))
    rel_error = (exact_results - rss_results).norm() / (exact_results.norm() + eps)
    
    if verbose: print(f'--> Relative error: {rel_error:.2e}')
    
    # Accuracy
    _, exact_preds = torch.max(exact_results, 1)
    _, rss_preds = torch.max(rss_results, 1)
    acc_diff = (rss_preds != exact_preds).float().mean().item()
    
    if verbose: print(f'--> Diff. of accuracies: {acc_diff:.2e}')
    
    torch.save(
        cores_rss,
        os.path.join(results_dir,
                     f'{info["total_time"]:.2e}_{info["val_eps"]:.2e}_'
                     f'{rel_error:.2e}_{acc_diff:.2e}.pt'))


# Tensorize multiple times
# ========================

def multiple_tt_rss(n, n_features, phys_dim, bond_dim,
                    samples_size, sketch_size, verbose=False):
    for _ in range(n):
        tt_rss_tensorization(n_features=n_features,
                             phys_dim=phys_dim,
                             bond_dim=bond_dim,
                             samples_size=samples_size,
                             sketch_size=sketch_size)


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
              '\t--n')
        sys.exit()
      
    # Read options and arguments
    try:
        opts, args = getopt.getopt(argv[1:], 'h', ['help', 'n'])
    except getopt.GetoptError:
        print('Available options are:\n'
              '\t--help, -h\n'
              '\t--n')
        sys.exit(2)
    
    # Save selected options
    options = {'n': False}
    
    for opt, arg in opts:
        if (opt == '-h') or (opt == '--help'):
            print('Available options are:\n'
                  '\t--help, -h\n'
                  '\t--n')
            sys.exit()
        elif opt == '--n':
            options['n'] = True
    
    # Multiple
    if options['n']:
        if len(args) < 6:
            print('In "n" mode the following arguments need '
                  'to be passed:\n'
                  '\t1) n\n'
                  '\t2) n_features\n'
                  '\t3) phys_dim\n'
                  '\t4) bond_dim\n'
                  '\t5) samples_size\n'
                  '\t6) sketch_size')
            sys.exit()
        else:
            n = int(args[0])
            n_features = int(args[1])
            phys_dim = int(args[2])
            bond_dim = int(args[3])
            samples_size = int(args[4])
            sketch_size = int(args[5])
        
        multiple_tt_rss(n=n,
                        n_features=n_features,
                        phys_dim=phys_dim,
                        bond_dim=bond_dim,
                        samples_size=samples_size,
                        sketch_size=sketch_size)
    
    else:
        if len(args) < 5:
            print('The following arguments need to be passed:\n'
                  '\t1) n_features\n'
                  '\t2) phys_dim\n'
                  '\t3) bond_dim\n'
                  '\t4) samples_size\n'
                  '\t5) sketch_size')
            sys.exit()
        else:
            n_features = int(args[0])
            phys_dim = int(args[1])
            bond_dim = int(args[2])
            samples_size = int(args[3])
            sketch_size = int(args[4])
        
        tt_rss_tensorization(n_features=n_features,
                             phys_dim=phys_dim,
                             bond_dim=bond_dim,
                             samples_size=samples_size,
                             sketch_size=sketch_size,
                             verbose=True)
