import os
import sys
import getopt

from math import log, sqrt

import torch

import tensorkrowch as tk
from tensorkrowch.decompositions import tt_rss


torch.set_num_threads(1)
cwd = os.getcwd()
eps = 1e-10


# Tensorize
# =========

def order_parameter(cores):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    mps = tk.models.MPS(tensors=cores)
    mps.parameterize(set_param=False, override=True)
    
    A_plus = torch.tensor([[0, sqrt(2 / 3)],
                           [0, 0]])
    A_zero = torch.tensor([[-sqrt(1 / 3), 0],
                           [0, sqrt(1 / 3)]])
    A_minus = torch.tensor([[0, 0],
                            [-sqrt(2 / 3), 0]])
    aklt_core = torch.stack([A_plus, A_zero, A_minus], dim=1).to(torch.complex64)
    bond_dim = aklt_core.shape[0]
    
    sigma_x = torch.tensor([[0, sqrt(1 / 3)],
                            [sqrt(1 / 3), 0]])
    sigma_y = torch.tensor([[0, -sqrt(1 / 3)*1j],
                            [sqrt(1 / 3)*1j, 0]])
    sigma_z = torch.tensor([[sqrt(1 / 3), 0],
                            [0, -sqrt(1 / 3)]])
    aklt_core_pauli = torch.stack([sigma_x, sigma_y, sigma_z], dim=1)
    
    U = torch.linalg.lstsq(
        aklt_core.permute(0, 2, 1).reshape(4, 3).to(aklt_core_pauli.dtype),
        aklt_core_pauli.permute(0, 2, 1).reshape(4, 3)).solution
    
    
    # Blocking
    L = 10
    mps_copy = mps.copy(share_tensors=True)
    mps_copy.left_node.move_to_network(mps)

    ux = torch.Tensor([[1, 0, 0],
                       [0, -1, 0],
                       [0, 0, -1]]).to(torch.complex64)
    uz = torch.Tensor([[-1, 0, 0],
                       [0, -1, 0],
                       [0, 0, 1]]).to(torch.complex64)

    ux = (U @ ux @ U.H).to(device)
    uz = (U @ uz @ U.H).to(device)

    start = (n_features // 2) - (L * 5 // 2)
    node_blocks = [mps.mats_env[(start + i*L):(start + (i+1)*L)]
                   for i in range(5)]
    node_blocks_copy = [mps_copy.mats_env[(start + i*L):(start + (i+1)*L)]
                        for i in range(5)]
    
    for i in [1, 2]:
        for j, node in enumerate(node_blocks[i]):
            node_z = tk.Node(tensor=uz,
                             name=f'node_z_({i}_{j})',
                             axes_names=('input', 'output'),
                             network=mps)
            node['input'] ^ node_z['output']
    
    for i in [1, 2]:
        for j, node in enumerate(node_blocks_copy[i]):
            node_x = tk.Node(tensor=ux,
                             name=f'node_x_({i}_{j})',
                             axes_names=('input', 'output'),
                             network=mps)
            node['input'] ^ node_x['output']
            
    i1_lst = [0, 1, 2, 3, 4]
    i2_lst = [0, 3, 2, 1, 4]

    contracted_blocks = []
    for i1, i2 in zip(i1_lst, i2_lst):
        
        # Connect all nodes of each block
        for j in range(L):
            if i1 in [1, 2]:
                edge1 = node_blocks[i1][j].neighbours('input')['input']
            else:
                edge1 = node_blocks[i1][j]['input']
            
            if i2 in [1, 2]:
                edge2 = node_blocks_copy[i2][j].neighbours('input')['input']
            else:
                edge2 = node_blocks_copy[i2][j]['input']
                
            edge1 ^ edge2
        
        # Contract with ux, uz
        for j in range(L):
            if i1 in [1, 2]:
                node_blocks[i1][j] = node_blocks[i1][j]['input'].contract_()
            
            if i2 in [1, 2]:
                node_blocks_copy[i2][j] = node_blocks_copy[i2][j]['input'].contract_()
        
        # Contract each node of a block with the corresponding copy
        aux_results = []
        for j in range(L):
            aux_results.append(tk.contract_between_(node_blocks[i1][j],
                                                    node_blocks_copy[i2][j]))
        
        # Contract all nodes in each block in line
        result = aux_results[0]
        for j in range(1, L):
            result = tk.contract_between_(result, aux_results[j])
        
        contracted_blocks.append(result)
    
    _, left_node = contracted_blocks[0].split_(node1_axes=['left_0', 'left_1'],
                                               node2_axes=['right_0', 'right_1'],
                                               side='left',
                                               rank=1)

    right_node, _ = contracted_blocks[-1].split_(node1_axes=['left_0', 'left_1'],
                                                 node2_axes=['right_0', 'right_1'],
                                                 side='right',
                                                 rank=1)

    # If left_node/right_node is not semidefinite positive, we can choose other
    # left_node/right_node that is semidefinite positive, as it will always
    # exist. Therefore, if we see u and vh are positive multiples of -I, we can
    # multiply both by -1 to make them positive multiples of I
    
    if (left_node.tensor[0].float().diag() < 0).all():
        left_node.tensor = -1 * left_node.tensor
    
    if (right_node.tensor[..., 0].float().diag() < 0).all():
        right_node.tensor = -1 * right_node.tensor
    
    result = left_node @ contracted_blocks[1] @ contracted_blocks[2] @ \
        contracted_blocks[3] @ right_node
    
    op = result.tensor * bond_dim**2
    return op.float().item()


def tt_rss_tensorization(n_features, samples_size, sketch_size, verbose=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    results_dir = os.path.join(cwd, 'results', '4_interpretability', 'aklt')
    os.makedirs(results_dir, exist_ok=True)
    
    cores_dir = os.path.join(
        cwd, 'results', '4_interpretability', 'aklt',
        f'rss_{n_features}_{samples_size}_{sketch_size}')
    os.makedirs(cores_dir, exist_ok=True)
    
    # Create AKLT state
    A_plus = torch.tensor([[0, sqrt(2 / 3)],
                           [0, 0]])
    A_zero = torch.tensor([[-sqrt(1 / 3), 0],
                           [0, sqrt(1 / 3)]])
    A_minus = torch.tensor([[0, 0],
                            [-sqrt(2 / 3), 0]])

    aklt_core = torch.stack([A_plus, A_zero, A_minus], dim=1)
    
    phys_dim = aklt_core.shape[1]
    bond_dim = aklt_core.shape[0]
    
    boundary_conditions = [0, 0]
    aklt_cores = [aklt_core.to(device) for _ in range(n_features)]
    aklt_cores[0] = aklt_cores[0][boundary_conditions[0], :, :]
    aklt_cores[-1] = aklt_cores[-1][:, :, boundary_conditions[1]]
    
    # Sample configurations
    samples_file = f'samples_{n_features}_{samples_size}.pt'
    if not os.path.exists(os.path.join(results_dir, samples_file)):
        def aux_embedding(x):
            return tk.embeddings.basis(x, dim=phys_dim).float()
        
        mps = tk.models.MPS(tensors=aklt_cores)
        mps.parameterize(set_param=False, override=True)
        
        samples = torch.tensor([]).int()
        
        for i in range(n_features):
            mps.unset_data_nodes()
            mps.reset()
            mps.in_features = torch.arange(i + 1).tolist()
            
            new_feature = torch.arange(phys_dim).view(-1, 1)
            new_feature = new_feature.repeat(samples_size, 1)
            
            if i > 0:
                aux_samples = samples.repeat(1, phys_dim)
                aux_samples = aux_samples.reshape(samples_size * phys_dim, i)
            else:
                aux_samples = samples
            
            aux_samples = torch.cat([aux_samples, new_feature], dim=1)
            
            density = mps(aux_embedding(aux_samples.to(device)),
                          marginalize_output=True,
                          renormalize=True)
            
            if i == (n_features - 1):
                density = torch.outer(density, density)
            
            distr = density.diagonal().reshape(samples_size, phys_dim)
            distr = distr / distr.sum(dim=-1, keepdim=True)
            distr = distr.cumsum(dim=-1)
            
            probs = torch.rand(samples_size, 1).to(device)
            new_samples = phys_dim - (probs < distr).sum(dim=-1)
            
            if i > 0:
                samples = torch.cat([samples,
                                    new_samples.cpu().int().reshape(-1, 1)], dim=1)
            else:
                samples = new_samples.cpu().int().reshape(-1, 1)

        samples = samples / phys_dim
        torch.save(samples, os.path.join(results_dir, samples_file))
        
        mps.unset_data_nodes()
        mps.reset()
        mps.trace(torch.zeros(1, n_features, phys_dim, device=device))
    
    samples = torch.load(os.path.join(results_dir, samples_file),
                         weights_only=False)
    
    mps = tk.models.MPS(tensors=aklt_cores)
    
    def embedding(x):
        x = tk.embeddings.discretize(x, base=phys_dim, level=1).squeeze(-1).int()
        x = tk.embeddings.basis(x, dim=phys_dim).float() # batch x n_features x dim
        return x
    
    # Estimate scale
    mps.set_data_nodes()
    mps.add_data(embedding(samples.to(device)))
    
    log_scale = 0
    
    stack = tk.stack(mps.mats_env)
    stack_data = tk.stack([node.neighbours('input') for node in mps.in_env])
    stack ^ stack_data
    
    mats_results = tk.unbind(stack @ stack_data)

    mats_results[0] = mps.left_node @ mats_results[0]
    mats_results[-1] = mats_results[-1] @ mps.right_node

    result = mats_results[0]
    for mat in mats_results[1:]:
        result @= mat
        
        log_scale += result.norm().log()
        result = result.renormalize()
    
    scale = (-log_scale / n_features).exp()
    
    aklt_cores = [aklt_core * scale for aklt_core in aklt_cores]
    
    mps = tk.models.MPS(tensors=aklt_cores)
    mps.parameterize(set_param=False, override=True)
    mps.trace(torch.zeros(1, n_features, phys_dim, device=device))
    
    def fun(x): return mps(embedding(x)).unsqueeze(-1)
    
    domain = torch.arange(phys_dim).float() / phys_dim
    
    perm = torch.randperm(samples.size(0))
    sketch_samples = samples[perm][:sketch_size]
    
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
    mps_rss = tk.models.MPS(tensors=[c.to(device) for c in cores_rss])
    mps_rss.parameterize(set_param=False, override=True)
    mps_rss.trace(torch.zeros(1, n_features, phys_dim, device=device))
    
    exact_results = mps(embedding(samples.to(device)))
    rss_results = mps_rss(embedding(samples.to(device)))
    rel_error = (exact_results - rss_results).norm() / (exact_results.norm() + eps)
    
    if verbose: print(f'--> Relative error: {rel_error:.2e}')
    
    
    # Fidelity
    mps.reset()
    mps.unset_data_nodes()
    mps_rss.reset()
    mps_rss.unset_data_nodes()
    
    mps_norm = mps.norm(log_scale=True)
    mps_rss_norm = mps_rss.norm(log_scale=True)

    mps.reset()
    mps_rss.reset()
    
    for node1, node2 in zip(mps.mats_env, mps_rss.mats_env):
        node1['input'] ^ node2['input']
    
    log_scale = 0

    # Contract mps with mps_rss
    stack = tk.stack(mps.mats_env)
    stack_rss = tk.stack(mps_rss.mats_env)
    stack ^ stack_rss

    mats_results = tk.unbind(stack @ stack_rss)
    
    mats_results[0] = mps.left_node @ (mps_rss.left_node @ mats_results[0])
    mats_results[-1] = (mats_results[-1] @ mps.right_node) @ mps_rss.right_node

    result = mats_results[0]
    for mat in mats_results[1:]:
        result @= mat
        
        log_scale += result.norm().log()
        result = result.renormalize()

    approx_mps_norm = (result.tensor.log() + log_scale) / 2
    fidelity = (2*approx_mps_norm - mps_norm - mps_rss_norm).exp()
    
    if verbose: print(f'--> Fidelity: {fidelity:.2e}')
    
    
    # Order parameter
    aklt_cores = [aklt_core.to(torch.complex64) / scale
                  for aklt_core in aklt_cores]
    op = order_parameter(aklt_cores)
    
    mps_rss = tk.models.MPS(tensors=[c.to(torch.complex64).to(device)
                                     for c in cores_rss])
    mps_rss.canonicalize(oc=0, renormalize=True)
    
    # Renormalize cores
    log_norm = 0

    for node in mps_rss.mats_env:
        log_norm += (node.norm() / scale).log()
        node.tensor = node.tensor / node.norm()

    for node in mps_rss.mats_env[1:-1]:
        log_norm -= log(sqrt(2))
        node.tensor = node.tensor * sqrt(2)

    for node in mps_rss.mats_env[:1] + mps_rss.mats_env[-1:]:
        node.tensor = node.tensor * (log_norm / 2).exp()
    
    op_rss = order_parameter(mps_rss.tensors)
    
    if verbose: print(f'--> Order Parameter: {op:.2e}, {op_rss:.2e}')
    
    torch.save(
        cores_rss,
        os.path.join(cores_dir,
                     f'{info["total_time"]:.2e}_{info["val_eps"]:.2e}_'
                     f'{rel_error:.2e}_{fidelity:.2e}_{op:.2e}_{op_rss:.2e}.pt'))


# Tensorize multiple times
# ========================

def multiple_tt_rss(n, n_features, samples_size, sketch_size, verbose=False):
    for _ in range(n):
        tt_rss_tensorization(n_features=n_features,
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
        if len(args) < 4:
            print('In "n" mode the following arguments need '
                  'to be passed:\n'
                  '\t1) n\n'
                  '\t2) n_features\n'
                  '\t3) samples_size\n'
                  '\t4) sketch_size')
            sys.exit()
        else:
            n = int(args[0])
            n_features = int(args[1])
            samples_size = int(args[2])
            sketch_size = int(args[3])
        
        multiple_tt_rss(n=n,
                        n_features=n_features,
                        samples_size=samples_size,
                        sketch_size=sketch_size)
    
    else:
        if len(args) < 3:
            print('The following arguments need to be passed:\n'
                  '\t1) n_features\n'
                  '\t2) samples_size\n'
                  '\t3) sketch_size')
            sys.exit()
        else:
            n_features = int(args[0])
            samples_size = int(args[1])
            sketch_size = int(args[2])
        
        tt_rss_tensorization(n_features=n_features,
                             samples_size=samples_size,
                             sketch_size=sketch_size,
                             verbose=True)
