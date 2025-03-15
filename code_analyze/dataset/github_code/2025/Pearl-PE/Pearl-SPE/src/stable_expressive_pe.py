from typing import List, Callable

import torch
from torch import nn
from torch_geometric.utils import unbatch

from src.gin import GIN
from src.gine import GINE
from src.gin_deepsets import GINDeepsets
from src.ppgn import MaskedPPGN
from src.deepsets import DeepSets, MaskedDeepSets
from src.transformer import Transformer
from src.mlp import MLP
from src.utils import mask2d_sum_pooling, mask2d_diag_offdiag_meanpool

from torch_scatter import scatter_add
from torch_sparse import SparseTensor, matmul, fill_diag, sum, mul
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import add_self_loops,get_laplacian,remove_self_loops
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch.nn import Parameter
from src.ign import IGN2to1_mask
from src.schema import Schema
from torch_geometric.nn.conv import MessagePassing
from scipy.special import comb
import math

class SwiGLU(nn.Module):
    def __init__(self, input_dim):
        super(SwiGLU, self).__init__()
        self.fc1 = nn.Linear(input_dim, input_dim)
        self.fc2 = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        swish_part = self.fc1(x) * torch.sigmoid(self.fc1(x))  
        gate = torch.sigmoid(self.fc2(x))  # Sigmoid
        return swish_part * gate 
    

def filter1(S, W, k):
    # S is laplacian and W is NxN e or NxM x_m
    out = W
    w_list = []
    w_list.append(out.unsqueeze(-1))
    for i in range(k-1): 
        out = S @ out # NxN or NxM
        w_list.append(out.unsqueeze(-1)) 
    return torch.cat(w_list, dim=-1) #NxMxK

def bern_filter(S, W, k):
    out = W
    w_list = []
    w_list.append(out.unsqueeze(-1))
    for i in range(1, k): 
        L = (1/(2**k)) * math.comb(k, i) * torch.linalg.matrix_power(
                                    (2*(torch.eye(S.shape[0]).to(S.device)) - S), k) @ S
        out = L @ W # NxN or NxM
        w_list.append(out.unsqueeze(-1)) 
    return torch.cat(w_list, dim=-1)

class StableExpressivePE(nn.Module):
    phi: nn.Module
    psi_list: nn.ModuleList
    def __init__(self, phi: nn.Module, psi_list: List[nn.Module], BASIS, k=16, mlp_nlayers=1, mlp_hid=16, spe_act='relu', mlp_out=16) -> None:
        super().__init__()
        #out_dim = len(psi_list)
        print("In spe mlp using activation: ", spe_act)
        self.mlp_nlayers = mlp_nlayers
        if mlp_nlayers > 0:
            if mlp_nlayers == 1:
                assert(mlp_hid == mlp_out)
            self.bn = nn.ModuleList()
            self.mlp_nlayers = mlp_nlayers
            self.layers = nn.ModuleList([nn.Linear(k if i==0 else mlp_hid, 
                                        mlp_hid if i<mlp_nlayers-1 else mlp_out, bias=True) for i in range(mlp_nlayers)])
            self.norms = nn.ModuleList([nn.BatchNorm1d(mlp_hid if i<mlp_nlayers-1 else mlp_out,track_running_stats=True) for i in range(mlp_nlayers)])
            #for i in range(mlp_nlayers - 1):
            #    self.layers.append(nn.Linear(k, mlp_hid, bias=True))
            #    self.bn.append(nn.BatchNorm1d(mlp_hid))
            #self.layers.append(nn.Linear(mlp_hid, mlp_hid, bias=True))
            #self.bn.append(nn.BatchNorm1d(mlp_hid))
        if spe_act == 'relu':
            self.activation = nn.ReLU(inplace=False)
        elif spe_act == "swish":
            self.activation = nn.SiLU()
        else:
            self.activation = SwiGLU(mlp_hid) ## edit if you want more than 1 mlp layers!!
        self.phi = phi
        self.k = k
        self.BASIS = BASIS
        print("SPE BASIS IS: ", self.BASIS)
        print("SPE k is: ", self.k)

    def forward(
        self, Lap, W, edge_index: torch.Tensor, batch: torch.Tensor, final=False
    ) -> torch.Tensor:
        """
        :param Lap: Laplacian
        :param W: B*[NxM] or BxNxN
        :param edge_index: Graph connectivity in COO format. [2, E_sum]
        :param batch: Batch index vector. [N_sum]
        :return: Positional encoding matrix. [N_sum, D_pe]
        """
        W_list = []
        # for loop N times for each Nx1 e
        if isinstance(W[0], int):
            print("WRONG INSTANCE")
            # split into N*B*[Nx1]
            j = 0
            for lap, w in zip(Lap, W):
                for i in range(w):
                    e_i = torch.zeros(w).to(device)
                    e_i[i] = 1
                    output = filter1(lap, e_i, self.k)  #
                    #output = bern_filter(lap, e_i, self.k) # # output [NxMxK]
                    '''if self.mlp_nlayers > 0:
                        for layer, bn in zip(self.layers, self.bn):
                            output = output.transpose(0, 1)     # MxNxK
                            output = layer(output)
                            output = bn(output.transpose(1,2)).transpose(1,2)   # BN of MxKxN
                            output = self.relu(output)
                            output = output.transpose(0, 1)'''
                    W_list.append(output)             # [NxMxK]*B
                if j == 0:
                    out = self.phi(W_list, edge_index, self.BASIS)
                else:
                    out += self.phi(W_list, edge_index, self.BASIS)
                j += 1
            return out
        else:
            for lap, w in zip(Lap, W):
                output = filter1(lap, w, self.k)   # output [NxMxK]
                #output = bern_filter(lap, w, self.k) #filter1(lap, w, self.k)
                if self.mlp_nlayers > 0:
                    for layer, bn in zip(self.layers, self.norms):
                        output = output.transpose(0, 1)
                        output = layer(output)
                        output = bn(output.transpose(1,2)).transpose(1,2)
                        output = self.activation(output)
                        output = output.transpose(0, 1)
                W_list.append(output)             # [NxMxK]*B
            return self.phi(W_list, edge_index, self.BASIS, final=final)   # [N_sum, D_pe]

    @property
    def out_dims(self) -> int:
        return self.phi.out_dims
    



class MaskedStableExpressivePE(nn.Module):
    phi: nn.Module
    psi_list: nn.ModuleList

    def __init__(self, phi: nn.Module, psi_list: List[nn.Module]) -> None:
        super().__init__()
        self.phi = phi
        self.psi_list = nn.ModuleList(psi_list)

    def forward(
        self, Lambda: torch.Tensor, V: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor
    ) -> torch.Tensor:
        """
        :param Lambda: Eigenvalue vectors. [B, D_pe]
        :param V: Concatenated eigenvector matrices. [N_sum, D_pe]
        :param edge_index: Graph connectivity in COO format. [2, E_sum]
        :param batch: Batch index vector. [N_sum]
        :return: Positional encoding matrix. [N_sum, D_pe]
        """
        Lambda = Lambda.unsqueeze(dim=2)   # [B, D_pe, 1]
        # Lambda = torch.cat([torch.cat([torch.cos(Lambda / 10000**(i/8)), torch.sin(Lambda / 10000**(i / 8))], dim=-1)
                            # for i in range(16)], dim=-1)
        a = torch.arange(0, Lambda.size(1)).unsqueeze(0).to(Lambda.device)
        mask = torch.cat([a < torch.sum(batch == i) for i in range(batch[-1]+1)], dim=0) # [B, D_pe, 1]
        Z = torch.stack([
            psi(Lambda, mask.unsqueeze(-1)).squeeze(dim=2)     # [B, D_pe]
            for psi in self.psi_list
        ], dim=2)                          # [B, D_pe, M]

        V_list = unbatch(V, batch, dim=0)   # [N_i, D_pe] * B
        Z_list = list(Z)                    # [D_pe, M] * B

        W_list = []                        # [N_i, N_i, M] * B
        for V, Z in zip(V_list, Z_list):   # [N_i, D_pe] and [D_pe, M]
            V = V.unsqueeze(dim=0)         # [1, N_i, D_pe]
            Z = Z.permute(1, 0)            # [M, D_pe]
            Z = Z.diag_embed()             # [M, D_pe, D_pe]
            V_T = V.mT                     # [1, D_pe, N_i]
            W = V.matmul(Z).matmul(V_T)    # [M, N_i, N_i]
            W = W.permute(1, 2, 0)         # [N_i, N_i, M]
            W_list.append(W)

        return self.phi(W_list, edge_index)   # [N_sum, D_pe]

    @property
    def out_dims(self) -> int:
        return self.phi.out_dims



class MLPPhi(nn.Module):
    gin: GIN

    def __init__(
            self, n_layers: int, in_dims: int, hidden_dims: int, out_dims: int, create_mlp: Callable[[int, int], MLP]
    ) -> None:
        super().__init__()
        # self.mlp = MLP(n_layers, in_dims, hidden_dims, out_dims, use_bn=False, activation='relu', dropout_prob=0.0)
        test_mlp = create_mlp(1, 1)
        use_bn, dropout_prob = test_mlp.layers[0].bn is not None, test_mlp.dropout.p
        self.mlp = MLP(n_layers, in_dims, hidden_dims, out_dims, use_bn=use_bn, activation='relu',
                       dropout_prob=dropout_prob, norm_type="layer")
        del test_mlp

    def forward(self, W_list: List[torch.Tensor], edge_index: torch.Tensor) -> torch.Tensor:
        """
        :param W_list: The {V * psi_l(Lambda) * V^T: l in [m]} tensors. [N_i, N_i, M] * B
        :param edge_index: Graph connectivity in COO format. [2, E_sum]
        :return: Positional encoding matrix. [N_sum, D_pe]
        """
        n_max = max(W.size(0) for W in W_list)
        W_pad_list = []     # [N_i, N_max, M] * B
        mask = [] # node masking, [N_i, N_max] * B
        for W in W_list:
            zeros = torch.zeros(W.size(0), n_max - W.size(1), W.size(2), device=W.device)
            W_pad = torch.cat([W, zeros], dim=1)   # [N_i, N_max, M]
            W_pad_list.append(W_pad)
            mask.append((torch.arange(n_max, device=W.device) < W.size(0)).tile((W.size(0), 1))) # [N_i, N_max]

        W = torch.cat(W_pad_list, dim=0)   # [N_sum, N_max, M]
        mask = torch.cat(mask, dim=0)   # [N_sum, N_max]
        PE = self.mlp(W)       # [N_sum, N_max, D_pe]
        return (PE * mask.unsqueeze(-1)).sum(dim=1)               # [N_sum, D_pe]
        # return PE.sum(dim=1)

    @property
    def out_dims(self) -> int:
        return self.mlp.out_dims



class GINPhi(nn.Module):
    gin: GIN

    def __init__(
        self, n_layers: int, in_dims: int, hidden_dims: int, out_dims: int, create_mlp: Callable[[int, int], MLP], bn: bool, RAND_LAP
    ) -> None:
        super().__init__()
        self.gin = GIN(n_layers, in_dims, hidden_dims, out_dims, create_mlp, bn, laplacian=RAND_LAP)
        self.mlp = create_mlp(out_dims, out_dims, use_bias=True)
        self.running_sum = 0

    def forward(self, W_list: List[torch.Tensor], edge_index: torch.Tensor, BASIS, mean=False, final=True) -> torch.Tensor:
        """
        :param W_list: The {V * psi_l(Lambda) * V^T: l in [m]} tensors. [N_i, N_i, M] * B
        :param edge_index: Graph connectivity in COO format. [2, E_sum]
        :return: Positional encoding matrix. [N_sum, D_pe]
        """ 
        if not BASIS:
            W = torch.cat(W_list, dim=0)   # [N_sum, M, K]
            PE = self.gin(W, edge_index)  
            if mean:
                PE = (PE).mean(dim=1) # sum or mean along M? get N, D_pe
            else:
                PE = (PE).sum(dim=1)
                self.running_sum += PE
            if final:
                PE = self.running_sum
                self.running_sum = 0
            return PE              # [N_sum, D_pe]
        else:
            n_max = max(W.size(0) for W in W_list)
            W_pad_list = []     # [N_i, N_max, M] * B
            mask = [] # node masking, [N_i, N_max] * B
            for W in W_list:
                zeros = torch.zeros(W.size(0), n_max - W.size(1), W.size(2), device=W.device)
                W_pad = torch.cat([W, zeros], dim=1)   # [N_i, N_max, M]
                W_pad_list.append(W_pad)
                mask.append((torch.arange(n_max, device=W.device) < W.size(0)).tile((W.size(0), 1))) # [N_i, N_max]
            W = torch.cat(W_pad_list, dim=0)   # [N_sum, N_max, M]
            mask = torch.cat(mask, dim=0)   # [N_sum, N_max]
            PE = self.gin(W, edge_index, mask=mask)       # [N_sum, N_max, D_pe]
            PE = (PE * mask.unsqueeze(-1)).sum(dim=1)
            return PE
        # return PE.sum(dim=1)

    @property
    def out_dims(self) -> int:
        return self.gin.out_dims


class GINEPhi(nn.Module):
    gine: GINE

    def __init__(
            self, n_layers: int, in_dims: int, hidden_dims: int, out_dims: int, create_mlp: Callable[[int, int], MLP]
    ) -> None:
        super().__init__()
        self.gine = GINE(n_layers, in_dims, hidden_dims, out_dims, create_mlp)

    def forward(self, W_list: List[torch.Tensor], edge_index: torch.Tensor) -> torch.Tensor:
        """
        :param W_list: The {V * psi_l(Lambda) * V^T: l in [m]} tensors. [N_i, N_i, M] * B
        :param edge_index: Graph connectivity in COO format. [2, E_sum]
        :return: Positional encoding matrix. [N_sum, D_pe]
        """
        n_max = max(W.size(0) for W in W_list)
        W_pad_list = []                            # [N_i, N_max, M] * B
        for W in W_list:
            zeros = torch.zeros(W.size(0), n_max - W.size(1), W.size(2), device=W.device)
            W_pad = torch.cat([W, zeros], dim=1)   # [N_i, N_max, M]
            W_pad_list.append(W_pad)

        W = torch.cat(W_pad_list, dim=0)   # [N_sum, N_max, M]
        PE = self.gin(W, edge_index)       # [N_sum, N_max, D_pe]
        return PE.sum(dim=1)               # [N_sum, D_pe]

    @property
    def out_dims(self) -> int:
        return self.gin.out_dims


class PPGNPhi(nn.Module):
    ppgn: MaskedPPGN

    def __init__(self, n_layers: int, in_dims: int, hidden_dims: int, out_dims: int,
                 create_mlp: Callable[[int, int], MLP]) -> None:
        super(PPGNPhi, self).__init__()
        self.ppgn = MaskedPPGN(in_dims, hidden_dims, out_dims, create_mlp, num_rb_layer=n_layers)
        self.pe_project = nn.Linear(2*out_dims, out_dims)

    def forward(self, W_list: List[torch.Tensor], edge_index: torch.Tensor) -> torch.Tensor:
        """
        :param W_list: The {V * psi_l(Lambda) * V^T: l in [m]} tensors. [N_i, N_i, M] * B
        :param edge_index: Graph connectivity in COO format. [2, E_sum]
        :return: Positional encoding matrix. [N_sum, D_pe]
        """
        # No edge info incorporated currently, TO DO: incorporate edge info into W
        n_max = max(W.size(0) for W in W_list)
        W_pad_list = []  # [N_max, N_max, M] * B
        mask = []
        for W in W_list:
            zeros = torch.zeros(W.size(0), n_max - W.size(1), W.size(2), device=W.device)
            W_pad = torch.cat([W, zeros], dim=1)  # [N_i, N_max, M]
            zeros = torch.zeros(n_max - W_pad.size(0), W_pad.size(1), W_pad.size(2), device=W_pad.device)
            W_pad = torch.cat([W_pad, zeros], dim=0)  # [N_max, N_max, M]
            W_pad = torch.unsqueeze(W_pad, dim=0) # [1, N_max, N_max, M]
            W_pad_list.append(W_pad)
            mask.append((torch.arange(n_max, device=W.device) < W.size(0)).unsqueeze(0)) # [1, N_max]

        W = torch.cat(W_pad_list, dim=0)  # [B, N_max, N_max, M]
        mask = torch.cat(mask, dim=0) # [B, N_max]
        mask_2d = mask.float().unsqueeze(-1) # [B, N_max, 1]
        mask_2d = torch.matmul(mask_2d, mask_2d.transpose(1, 2)).unsqueeze(-1) # [B, N_max, N_max, 1]
        PE = self.ppgn(W, mask_2d)   # [B, N_max, N_max, D_pe]
        # PE = mask2d_sum_pooling(PE, mask_2d) # TO DO: more variants of pooling functions, e.g. diag/off-diag pooling
        PE = mask2d_diag_offdiag_meanpool(PE, mask_2d)
        PE = self.pe_project(PE)
        # PE = PE.sum(dim=1)
        PE = PE.view(-1, PE.size(-1))[mask.view(-1)] # [N_sum, D_pe]
        return PE

    @property
    def out_dims(self) -> int:
        return self.ppgn.out_dims


class GINDeepSetsPhi(nn.Module):
    """
    inspired by Vignac, Clement, Andreas Loukas, and Pascal Frossard.
    "Building powerful and equivariant graph neural networks with structural message-passing."
    Advances in neural information processing systems 33 (2020): 14143-14155.
    """
    def __init__(self, n_layers: int, in_dims: int, hidden_dims: int, out_dims: int,
                 create_mlp: Callable[[int, int], MLP]):
        super(GINDeepSetsPhi, self).__init__()
        self.gin_deepsets = GINDeepsets(n_layers, in_dims, hidden_dims, out_dims, create_mlp)

    def forward(self, W_list: List[torch.Tensor], edge_index: torch.Tensor) -> torch.Tensor:
        """
        :param W_list: The {V * psi_l(Lambda) * V^T: l in [m]} tensors. [N_i, N_i, M] * B
        :param edge_index: Graph connectivity in COO format. [2, E_sum]
        :return: Positional encoding matrix. [N_sum, D_pe]
        """
        n_max = max(W.size(0) for W in W_list)
        W_pad_list = []     # [N_i, N_max, M] * B
        mask = [] # node masking, [N_i, N_max] * B
        for W in W_list:
            zeros = torch.zeros(W.size(0), n_max - W.size(1), W.size(2), device=W.device)
            W_pad = torch.cat([W, zeros], dim=1)   # [N_i, N_max, M]
            W_pad_list.append(W_pad)
            mask.append((torch.arange(n_max, device=W.device) < W.size(0)).tile((W.size(0), 1))) # [N_i, N_max]

        W = torch.cat(W_pad_list, dim=0)   # [N_sum, N_max, M]
        mask = torch.cat(mask, dim=0).unsqueeze(-1)   # [N_sum, N_max]
        PE = self.gin_deepsets(W, edge_index, mask)       # [N_sum, N_max, D_pe]
        return (PE * mask).sum(dim=1)               # [N_sum, D_pe]

    @property
    def out_dims(self) -> int:
        return self.gin_deepsets.out_dims


class IGNPhi(nn.Module):
    def __init__(self, n_layers: int, in_dims: int, hidden_dims: int, out_dims: int,
                 create_mlp: Callable[[int, int], MLP], device):
        super(IGNPhi, self).__init__()
        self.ign = IGN2to1_mask(in_dims, hidden_dims, out_dims, num_layers=n_layers, device=device)
        self.out_dim = out_dims

    def forward(self, W_list: List[torch.Tensor], edge_index: torch.Tensor) -> torch.Tensor:
        """
        :param W_list: The {V * psi_l(Lambda) * V^T: l in [m]} tensors. [N_i, N_i, M] * B
        :param edge_index: Graph connectivity in COO format. [2, E_sum]
        :return: Positional encoding matrix. [N_sum, D_pe]
        """
        # No edge info incorporated currently, TO DO: incorporate edge info into W
        n_max = max(W.size(0) for W in W_list)
        W_pad_list = []  # [N_max, N_max, M] * B
        mask = []
        for W in W_list:
            zeros = torch.zeros(W.size(0), n_max - W.size(1), W.size(2), device=W.device)
            W_pad = torch.cat([W, zeros], dim=1)  # [N_i, N_max, M]
            zeros = torch.zeros(n_max - W_pad.size(0), W_pad.size(1), W_pad.size(2), device=W_pad.device)
            W_pad = torch.cat([W_pad, zeros], dim=0)  # [N_max, N_max, M]
            W_pad = torch.unsqueeze(W_pad, dim=0)  # [1, N_max, N_max, M]
            W_pad_list.append(W_pad)
            mask.append((torch.arange(n_max, device=W.device) < W.size(0)).unsqueeze(0))  # [1, N_max]

        W = torch.cat(W_pad_list, dim=0)  # [B, N_max, N_max, M]
        W = torch.transpose(W, 1, -1) # [B, M, N_max, N_max]
        mask = torch.cat(mask, dim=0)  # [B, N_max]
        mask_2d = mask.float().unsqueeze(-1)  # [B, N_max, 1]
        mask_2d = torch.matmul(mask_2d, mask_2d.transpose(1, 2)).unsqueeze(-1)  # [B, N_max, N_max, 1]
        PE = self.ign(W, mask).transpose(1, 2)  # [B, N_max, D_pe]
        # PE = mask2d_sum_pooling(PE, mask_2d) # TO DO: more variants of pooling functions, e.g. diag/off-diag pooling
        #PE = mask2d_diag_offdiag_meanpool(PE, mask_2d)
        #PE = self.pe_project(PE)
        # PE = PE.sum(dim=1)
        PE = PE.reshape(-1, PE.size(-1))[mask.view(-1)]  # [N_sum, D_pe]
        return PE

    @property
    def out_dims(self) -> int:
        return self.out_dim

class ZeroPsi(nn.Module):
    # this model will ignore any input and return non-informative output
    def __init__(self):
        super(ZeroPsi, self).__init__()

    def forward(self, x):
        return torch.ones_like(x).to(x.device)

class ZeroPhi(nn.Module):
    def __init__(self, out_dims):
        super(ZeroPhi, self).__init__()
        self.out_dims = out_dims

    def forward(self, W_list: List[torch.Tensor], edge_index: torch.Tensor):
        n_sum = 0
        for W in W_list:
            n_sum += W.size(0)
        return torch.zeros([n_sum, self.out_dims]).to(W_list[0].device)


def GetPhi(cfg: Schema, create_mlp: Callable[[int, int], MLP], device):
    if cfg.phi_model_name == 'gin':
        return GINPhi(cfg.n_phi_layers, cfg.RAND_mlp_out, cfg.phi_hidden_dims, cfg.pe_dims,
                                         create_mlp, cfg.batch_norm, RAND_LAP=cfg.RAND_LAP)
        #return GINPhi(cfg.n_phi_layers, cfg.n_psis, cfg.phi_hidden_dims, cfg.pe_dims,
                      #create_mlp) # no bn for padding input
    elif cfg.phi_model_name == 'gin_deepsets':
        return GINDeepSetsPhi(cfg.n_phi_layers, cfg.n_psis, cfg.phi_hidden_dims, cfg.pe_dims,
                                         create_mlp)
    elif cfg.phi_model_name == 'ppgn': # unstable now
        return PPGNPhi(cfg.n_phi_layers, cfg.n_psis, cfg.phi_hidden_dims, cfg.pe_dims,
                                         create_mlp)
    elif cfg.phi_model_name == 'ign':
        return IGNPhi(cfg.n_phi_layers, cfg.n_psis, cfg.phi_hidden_dims, cfg.pe_dims,
                       create_mlp, device)
    elif cfg.phi_model_name == 'mlp':
        return MLPPhi(cfg.n_phi_layers, cfg.n_psis, cfg.phi_hidden_dims, cfg.pe_dims, create_mlp)
    elif cfg.phi_model_name == 'zero':
        return ZeroPhi(cfg.pe_dims)
    else:
        raise Exception ("Phi function not implemented!")


def GetPsi(cfg: Schema):
    return [0] * cfg.n_psis
    '''if cfg.psi_model_name == 'deepsets':
        if cfg.pe_method.startswith("masked"):
            return MaskedDeepSets(cfg.n_psi_layers, 1, cfg.psi_hidden_dims, 1, cfg.psi_activation)
        else:
            return DeepSets(cfg.n_psi_layers, 1, cfg.psi_hidden_dims, 1, cfg.psi_activation)
    elif cfg.psi_model_name == 'transformer':
        return Transformer(cfg.n_psi_layers, 1, cfg.psi_hidden_dims, 1, cfg.num_heads)
    elif cfg.psi_model_name == 'mlp':
        return MLP(cfg.n_psi_layers, 1, cfg.psi_hidden_dims, 1, use_bn=cfg.mlp_use_bn, activation=cfg.psi_activation,
                   dropout_prob=0.0)
    elif cfg.psi_model_name == 'zero':
        return ZeroPsi()
    else:
        raise Exception ('Psi function not implemented!')'''