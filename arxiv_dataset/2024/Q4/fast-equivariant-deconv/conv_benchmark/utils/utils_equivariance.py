from .laplacian import prepare_laplacian, scipy_csr_to_sparse_tensor
from .sh_matrix import _sh_matrix
from utils.spherehealpix import SphereHealpix
import numpy as np 
import torch

def get_raw_sampling_hp(sampling_param, legacy=False, hemisphere=False):
    G = SphereHealpix(sampling_param, nest=True, k=8) # Construct Healpix Graph at resolution n_side
    G.compute_laplacian('normalized') # Compute Healpix laplacian
    laplacian = prepare_laplacian(G.L) # Get Healpix laplacian
    coords = G.coords
    if hemisphere:
        eps = 1e-10
        index_north_hemi = (G.coords[:, 2]>eps) + ((G.coords[:, 2]<eps)*(G.coords[:, 2]>-eps)*( G.coords[:, 1]>eps)) + ((G.coords[:, 2]<eps)*(G.coords[:, 2]>-eps)*(G.coords[:, 1]<eps)*(G.coords[:, 1]>-eps)*(G.coords[:, 0]>eps))
        distance = G.coords.dot(G.coords.T)
        indx = np.arange(distance.shape[0])[None].repeat(distance.shape[0], axis=0)
        sel = distance < -1+1e-5
        assert np.sum(np.sum(sel, axis=-1) != 1) == 0
        match_sel = indx[sel]
        laplacian = laplacian[index_north_hemi][:, index_north_hemi] + laplacian[index_north_hemi][:, match_sel][:, index_north_hemi]
        coords = coords[index_north_hemi]
    laplacian = scipy_csr_to_sparse_tensor(laplacian)
    return torch.Tensor(coords), laplacian

def get_sh_matrices(sh_degree, vec, symmetric):
    # Create the corresponding frequency threshold to smooth the signal
    # for exact equivariant rotation/interpolation
    assert int((sh_degree + 1) * (sh_degree/(1+symmetric) + 1)) <= vec.shape[0]
    S2SH, SH2S = _sh_matrix(sh_degree, vec.cpu().numpy(), symmetric=symmetric)
    S2SH, SH2S = torch.Tensor(S2SH), torch.Tensor(SH2S)
    return S2SH, SH2S


def invariant_attr_r3s2_fiber_bundle(pos, ori_grid, edge_index):
    pos = torch.Tensor(pos)
    ori_grid = torch.Tensor(ori_grid)
    
    pos_send, pos_receive = pos[edge_index[0]], pos[edge_index[1]]                # [num_edges, 3]
    rel_pos = (pos_send - pos_receive)                                            # [num_edges, 3]

    # Convenient shape
    rel_pos = rel_pos[:, None, :]                                                 # [num_edges, 1, 3]
    ori_grid_a = ori_grid[None,:,:]                                               # [1, num_ori, 3]
    ori_grid_b = ori_grid[:, None,:]                                              # [num_ori, 1, 3]

    invariant1 = (rel_pos * ori_grid_a).sum(dim=-1, keepdim=True)                 # [num_edges, num_ori, 1]
    invariant2 = (rel_pos - invariant1 * ori_grid_a).norm(dim=-1, keepdim=True)   # [num_edges, num_ori, 1]
    invariant3 = (ori_grid_a * ori_grid_b).sum(dim=-1, keepdim=True)              # [num_ori, num_ori, 1]
    
    # Note: We could apply the acos = pi/2 - asin, which is differentiable at -1 and 1
    # But found that this mapping is unnecessary as it is monotonic and mostly linear 
    # anyway, except close to -1 and 1. Not applying the arccos worked just as well.
    # invariant3 = torch.pi / 2 - torch.asin(invariant3.clamp(-1.,1.))
    return torch.cat([invariant1, invariant2],dim=-1), invariant3   