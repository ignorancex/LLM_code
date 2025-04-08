import torch
import torch.nn as nn

from module.nn.base import Dense
from module.nn.cfconv import CFConv
from module.nn.rbf import Gaussian_Smearing

class C3Net_Interaction(nn.Module):
    def __init__(self, n_atom_basis, n_spatial_basis, n_filters):
        super(C3Net_Interaction, self).__init__()
        
        self.filter_network = nn.Sequential(Dense(n_spatial_basis, n_filters, activation=True), Dense(n_filters, n_filters))

        self.cfconv = CFConv(n_atom_basis, n_filters, n_atom_basis, self.filter_network,activation=True)       # dense layer
        self.dense = Dense(n_atom_basis, n_atom_basis, bias=True, activation=None)

    def forward(self, s, neighbor_mask, neighbors, f_ij):
        # continuous-filter convolution interaction block followed by Dense layer
        v = self.cfconv(s, neighbor_mask, neighbors, f_ij)
        v = self.dense(v)
        return v


class C3Net(nn.Module):
    def __init__(self, n_atom_basis=16, cutoff=1.0, n_gaussians=16, n_filters=64):
        super(C3Net, self).__init__()

        self.embedding = nn.Embedding(18, n_atom_basis, padding_idx=0) #covalent bond lookup table
        
        self.atom_dense_network = Dense(n_atom_basis, n_filters, bias=True, activation=None)
        self.bond_dense_network = Dense(n_atom_basis, n_filters, bias=True, activation=None)

        # layer for expanding interatomic distances or geometry in a basis
        self.atom_distance_expansion1 = Gaussian_Smearing(0., cutoff, n_gaussians)
        self.atom_distance_expansion2 = Gaussian_Smearing(0., cutoff, n_gaussians)
        self.atom_distance_expansion3 = Gaussian_Smearing(0., cutoff, n_gaussians)
        
        # layer for expanding interatomic distances in a basis
        self.atom_interactions1 = C3Net_Interaction(n_atom_basis=n_filters,n_spatial_basis=n_gaussians,n_filters=n_filters)
        self.atom_interactions2 = C3Net_Interaction(n_atom_basis=n_filters,n_spatial_basis=n_gaussians,n_filters=n_filters)
        self.atom_interactions3 = C3Net_Interaction(n_atom_basis=n_filters,n_spatial_basis=n_gaussians,n_filters=n_filters)
        
        self.bond_interaction = C3Net_Interaction(n_atom_basis=n_filters,n_spatial_basis=n_filters,n_filters=n_filters)

        self.solv_network = nn.Sequential(Dense(5, n_atom_basis, activation=True), Dense(n_atom_basis, n_filters, activation=None))

        self.logP_prop = nn.Parameter(torch.tensor([-0.044473435,0.402331736,-0.078766874,0.416642071,0.276510845]))
        self.PAMPA = nn.Parameter(torch.tensor([-0.044473435,0.402331736,-0.078766874,0.416642071,0.276510845]))
        
    def forward(self, inputs):
        # get tensors from input dictionary
        atom_x = inputs['Encoder'] 
        atom_index_bond = inputs['Neighbor_Index_Bond'] #list of neighbor atoms
        mask_bond = inputs['Mask_Bond']
        mask_nonbond = inputs['Mask_NonBond']
        bond = inputs['Neighbor_Type_Bond']
        atom_r_ij = inputs['Neighbor_Distance_NonBond']
        
        bond_x = self.embedding(bond)
        atom = self.atom_dense_network(atom_x)
        bond = self.bond_dense_network(bond_x)
        
        for i in range(3):
            v = self.bond_interaction(atom, mask_bond, atom_index_bond, f_ij=bond)
            atom = atom+v

        solvent_prop = inputs["Solvent_Properties"]
        if 103 in inputs['Solvent_ID']:
            for i in range(inputs["Solvent_Properties"].size()[0]):
                if inputs['Solvent_ID'][i] == 103:
                    solvent_prop[i] = self.logP_prop
        
        if 104 in inputs['Solvent_ID']:
            for i in range(inputs["Solvent_Properties"].size()[0]):
                if inputs['Solvent_ID'][i] == 104:
                    
                    solvent_prop[i] = self.PAMPA
        
        solv = self.solv_network(solvent_prop)

        v1 = self.atom_interactions1(solv, mask_nonbond, None, f_ij=self.atom_distance_expansion1(atom_r_ij))
        
        v2 = self.atom_interactions2(solv, mask_nonbond, None, f_ij=self.atom_distance_expansion2(atom_r_ij))

        v3 = self.atom_interactions3(solv, mask_nonbond, None, f_ij=self.atom_distance_expansion3(atom_r_ij))

        v = v1+v2+v3
        atom_x =atom*v
        return atom_x
