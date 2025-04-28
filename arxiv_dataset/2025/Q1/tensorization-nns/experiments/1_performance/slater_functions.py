import os
import sys
import getopt
from functools import partial

import torch
import tntorch as tn

import tensorkrowch as tk
from tensorkrowch.decompositions import tt_rss


torch.set_num_threads(1)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cwd = os.getcwd()
eps = 1e-10


# Define Slater functions
# =======================

def slater(x, L, m, l):
    """
    Returns the output of the m-dimensional Slater function with input in the
    domain [0, L]^m.

    Parameters
    ----------
    x : torch.Tensor
        Tensor of binary inputs with shape batch x (d*m).
    L : int, float
        Limit of domain.
    m : int
        Number of dimensions/variables
    l : int
        Discretization level of each variable.
    """
    assert x.size(1) == l*m
    
    x = x.reshape(-1, l, m)
    
    total_x = torch.zeros_like(x)[:, 0, :].float()  # batch x m
    total_x += sum(x[:, i - 1, :] * 2 ** (-i) for i in range(1, l + 1))
    
    total_x = total_x * L  # total_x in [0, L]^m
    
    norm_x = total_x.norm(p=2, dim=1)
    result = (-norm_x).exp() / (norm_x + eps)
    
    return result


# Tensorize
# =========

def tt_rss_tensorization(L, m, l, bond_dim, samples_size, sketch_size, verbose=False):
    results_dir = os.path.join(cwd, 'results', '1_performance', 'slater_functions',
                               f'rss_{L}_{m}_{l}_{bond_dim}_{samples_size}_{sketch_size}')
    os.makedirs(results_dir, exist_ok=True)
    
    n_features = l*m
    domain = [torch.arange(2) for _ in range(n_features)]

    # Create dataset to compare results and check accuracy
    def discretize(x):
        return tk.embeddings.discretize(x, level=l).reshape(-1, l*m).int()
    
    aux_eps = 1e-2
    samples = torch.rand(size=(samples_size, m)) * (1 - 2*aux_eps) + aux_eps
    samples = discretize(samples)

    exact_results = slater(x=samples, L=L, m=m, l=l).to(device)
    
    def aux_slater(x): return slater(x=x, L=L, m=m, l=l).unsqueeze(1)
    def embedding(x): return tk.embeddings.basis(x, dim=2).float()

    sketch_samples = samples[torch.randperm(samples_size)][:sketch_size]

    if verbose: print('* Starting TT-RSS tensorization...')
    cores_rss, info = tt_rss(function=aux_slater,
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
    mps = tk.models.MPS(tensors=[c.to(device) for c in cores_rss])
    mps.trace(torch.zeros(1, n_features, 2, device=device),
              inline_input=True,
              inline_mats=True)

    rss_results = mps(embedding(samples.to(device)))
    rel_error = (exact_results - rss_results).norm() / (exact_results.norm() + eps)
    
    if verbose: print(f'--> Relative error: {rel_error:.2e}')

    torch.save(
        cores_rss,
        os.path.join(results_dir,
                     f'{info["total_time"]:.2e}_{info["val_eps"]:.2e}_'
                     f'{rel_error:.2e}.pt'))


def tt_cross_tensorization(L, m, l, bond_dim, samples_size, verbose=False):
    results_dir = os.path.join(cwd, 'results', '1_performance', 'slater_functions',
                               f'cross_{L}_{m}_{l}_{bond_dim}_{samples_size}')
    os.makedirs(results_dir, exist_ok=True)
    
    n_features = l*m
    domain = [torch.arange(2, device=device)] * n_features

    # Create dataset to compare results and check accuracy
    def discretize(x):
        return tk.embeddings.discretize(x, level=l).reshape(-1, l*m).int()
    
    aux_eps = 1e-2
    samples = torch.rand(size=(samples_size, m)) * (1 - 2*aux_eps) + aux_eps
    samples = discretize(samples)

    exact_results = slater(x=samples, L=L, m=m, l=l).to(device)
    
    def embedding(x): return tk.embeddings.basis(x, dim=2).float()
    
    if verbose: print('* Starting TT-RSS tensorization...')
    tt_cross, info = tn.cross(function=partial(slater, L=L, m=m, l=l),
                              domain=domain,
                              device=device,
                              function_arg='matrix',
                              rmax=bond_dim,
                              max_iter=10,  # 25 by default
                              eps=1e-3,
                              verbose=verbose,
                              return_info=True)
    
    if verbose:
        print(f'* Tensorization finished in {info["total_time"]:.2e} seconds')
        print(f'--> Val. error: {info["val_eps"]:.2e}')
    
    # Relative error
    cores_cross = tt_cross.cores
    cores_cross[0] = cores_cross[0][0]
    cores_cross[-1] = cores_cross[-1][..., 0]
    
    mps = tk.models.MPS(tensors=[c.to(device) for c in cores_cross])
    mps.trace(torch.zeros(1, n_features, 2, device=device),
              inline_input=True,
              inline_mats=True)

    cross_results = mps(embedding(samples.to(device)))
    rel_error = (exact_results - cross_results).norm() / (exact_results.norm() + eps)
    
    if verbose: print(f'--> Relative error: {rel_error:.2e}')

    torch.save(
        cores_cross,
        os.path.join(results_dir, f'{info["total_time"]:.2e}_{rel_error:.2e}.pt'))


# Tensorize multiple times
# ========================

def multiple_tt_rss(n, L, m, l, bond_dim, samples_size, sketch_size):
    for _ in range(n):
        tt_rss_tensorization(L=L,
                             m=m,
                             l=l,
                             bond_dim=bond_dim,
                             samples_size=samples_size,
                             sketch_size=sketch_size)


def multiple_tt_cross(n, L, m, l, bond_dim, samples_size):
    for _ in range(n):
        tt_cross_tensorization(L=L,
                               m=m,
                               l=l,
                               bond_dim=bond_dim,
                               samples_size=samples_size)


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
              '\t--rss\n'
              '\t--cross\n'
              '\t--n')
        sys.exit()
      
    # Read options and arguments
    try:
        opts, args = getopt.getopt(argv[1:], 'h', ['help', 'rss', 'cross', 'n'])
    except getopt.GetoptError:
        print('Available options are:\n'
              '\t--help, -h\n'
              '\t--rss\n'
              '\t--cross\n'
              '\t--n')
        sys.exit(2)
    
    # Save selected options
    options = {'rss': False,
               'cross': False,
               'n': False}
    
    for opt, arg in opts:
        if (opt == '-h') or (opt == '--help'):
            print('Available options are:\n'
                  '\t--help, -h\n'
                  '\t--rss\n'
                  '\t--cross\n'
                  '\t--n')
            sys.exit()
        elif opt == '--rss':
            options['rss'] = True
        elif opt == '--cross':
            options['cross'] = True
        elif opt == '--n':
            options['n'] = True
    
    # Check if selected options are compatible
    if options['rss'] and options['cross']:
        print('Options "rss" and "cross" are incompatible')
        sys.exit()
    elif not (options['rss'] or options['cross']):
        print('One of the options "rss" and "cross" should be chosen')
        sys.exit()
    
    # Multiple
    if options['n']:  
        # RSS
        if options['rss']:
            if len(args) < 7:
                print('In "n" and "rss" mode the following arguments need '
                      'to be passed:\n'
                      '\t1) n\n'
                      '\t2) L\n'
                      '\t3) m\n'
                      '\t4) l\n'
                      '\t5) bond_dim\n'
                      '\t6) samples_size\n'
                      '\t7) sketch_size')
                sys.exit()
            else:
                n = int(args[0])
                L = float(args[1])
                m = int(args[2])
                l = int(args[3])
                bond_dim = int(args[4])
                samples_size = int(args[5])
                sketch_size = int(args[6])
            
            multiple_tt_rss(n=n,
                            L=L,
                            m=m,
                            l=l,
                            bond_dim=bond_dim,
                            samples_size=samples_size,
                            sketch_size=sketch_size)
        
        # CROSS
        elif options['cross']:
            if len(args) < 6:
                print('In "n" and "cross" mode the following arguments need '
                      'to be passed:\n'
                      '\t1) n\n'
                      '\t2) L\n'
                      '\t3) m\n'
                      '\t4) l\n'
                      '\t5) bond_dim\n'
                      '\t6) samples_size')
                sys.exit()
            else:
                n = int(args[0])
                L = float(args[1])
                m = int(args[2])
                l = int(args[3])
                bond_dim = int(args[4])
                samples_size = int(args[5])
            
            multiple_tt_cross(n=n,
                              L=L,
                              m=m,
                              l=l,
                              bond_dim=bond_dim,
                              samples_size=samples_size)
    
    else:
        # RSS
        if options['rss']:
            if len(args) < 6:
                print('In "rss" mode the following arguments need to be passed:\n'
                      '\t1) L\n'
                      '\t2) m\n'
                      '\t3) l\n'
                      '\t4) bond_dim\n'
                      '\t5) samples_size\n'
                      '\t6) sketch_size')
                sys.exit()
            else:
                L = float(args[0])
                m = int(args[1])
                l = int(args[2])
                bond_dim = int(args[3])
                samples_size = int(args[4])
                sketch_size = int(args[5])
            
            tt_rss_tensorization(L=L,
                                 m=m,
                                 l=l,
                                 bond_dim=bond_dim,
                                 samples_size=samples_size,
                                 sketch_size=sketch_size,
                                 verbose=True)
        
        # CROSS
        elif options['cross']:
            if len(args) < 5:
                print('In "cross" mode the following arguments need to be passed:\n'
                      '\t1) L\n'
                      '\t2) m\n'
                      '\t3) d\n'
                      '\t4) bond_dim\n'
                      '\t5) samples_size')
                sys.exit()
            else:
                L = float(args[0])
                m = int(args[1])
                l = int(args[2])
                bond_dim = int(args[3])
                samples_size = int(args[4])
            
            tt_cross_tensorization(L=L,
                                   m=m,
                                   l=l,
                                   bond_dim=bond_dim,
                                   samples_size=samples_size,
                                   verbose=True)
