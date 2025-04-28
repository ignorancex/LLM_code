import logging
logger = logging.getLogger(__name__)

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import argparse
import copy
import json
import numpy as np
import pickle

import torch

import rdkit.Chem as Chem
from rdkit.Chem import DataStructs, rdmolops
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem.rdMHFPFingerprint import MHFPEncoder
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

from sklearn.preprocessing import OneHotEncoder
from torch_geometric.data import Data
from transformers import BertTokenizer, T5Tokenizer

from .featurizer import MoleculeFeaturizer

from open_biomed.data.molecule import Molecule


def one_hot_encoding(x, allowable_set, encode_unknown=False):
    """One-hot encoding.
    """
    if encode_unknown and (allowable_set[-1] is not None):
        allowable_set.append(None)

    if encode_unknown and (x not in allowable_set):
        x = None

    return list(map(lambda s: x == s, allowable_set))

def safe_index(l, e):
    """
    Return index of element e in list l. If e is not present, return the last index
    """
    try:
        return l.index(e)
    except:
        return len(l) - 1

# Atom featurization: Borrowed from dgllife.utils.featurizers.py

def atom_type_one_hot(atom, allowable_set=None, encode_unknown=False):
    """One hot encoding for the type of an atom.
    """
    if allowable_set is None:
        allowable_set = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca',
                         'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn',
                         'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au',
                         'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb']
    return one_hot_encoding(atom.GetSymbol(), allowable_set, encode_unknown)

def atom_degree_one_hot(atom, allowable_set=None, encode_unknown=False):
    """One hot encoding for the degree of an atom.
    """
    if allowable_set is None:
        allowable_set = list(range(11))
    return one_hot_encoding(atom.GetDegree(), allowable_set, encode_unknown)

def atom_implicit_valence_one_hot(atom, allowable_set=None, encode_unknown=False):
    """One hot encoding for the implicit valence of an atom.
    """
    if allowable_set is None:
        allowable_set = list(range(7))
    return one_hot_encoding(atom.GetImplicitValence(), allowable_set, encode_unknown)

def atom_formal_charge(atom):
    """Get formal charge for an atom.
    """
    return [atom.GetFormalCharge()]

def atom_num_radical_electrons(atom):
    return [atom.GetNumRadicalElectrons()]

def atom_hybridization_one_hot(atom, allowable_set=None, encode_unknown=False):
    """One hot encoding for the hybridization of an atom.
    """
    if allowable_set is None:
        allowable_set = [Chem.rdchem.HybridizationType.SP,
                         Chem.rdchem.HybridizationType.SP2,
                         Chem.rdchem.HybridizationType.SP3,
                         Chem.rdchem.HybridizationType.SP3D,
                         Chem.rdchem.HybridizationType.SP3D2]
    return one_hot_encoding(atom.GetHybridization(), allowable_set, encode_unknown)

def atom_is_aromatic(atom):
    """Get whether the atom is aromatic.
    """
    return [atom.GetIsAromatic()]

def atom_total_num_H_one_hot(atom, allowable_set=None, encode_unknown=False):
    """One hot encoding for the total number of Hs of an atom.
    """
    if allowable_set is None:
        allowable_set = list(range(5))
    return one_hot_encoding(atom.GetTotalNumHs(), allowable_set, encode_unknown)

def atom_is_in_ring(atom):
    """Get whether the atom is in ring.
    """
    return [atom.IsInRing()]

def atom_chirality_type_one_hot(atom, allowable_set=None, encode_unknown=False):
    """One hot encoding for the chirality type of an atom.
    """
    if not atom.HasProp('_CIPCode'):
        return [False, False]

    if allowable_set is None:
        allowable_set = ['R', 'S']
    return one_hot_encoding(atom.GetProp('_CIPCode'), allowable_set, encode_unknown)

# Atom featurization: Borrowed from dgllife.utils.featurizers.py

def bond_type_one_hot(bond, allowable_set=None, encode_unknown=False):
    """One hot encoding for the type of a bond.
    """
    if allowable_set is None:
        allowable_set = [Chem.rdchem.BondType.SINGLE,
                         Chem.rdchem.BondType.DOUBLE,
                         Chem.rdchem.BondType.TRIPLE,
                         Chem.rdchem.BondType.AROMATIC]
    return one_hot_encoding(bond.GetBondType(), allowable_set, encode_unknown)

class MolOneHotFeaturizer(MoleculeFeaturizer):
    smiles_char = ['?', '#', '%', ')', '(', '+', '-', '.', '1', '0', '3', '2', '5', '4',
       '7', '6', '9', '8', '=', 'A', 'C', 'B', 'E', 'D', 'G', 'F', 'I',
       'H', 'K', 'M', 'L', 'O', 'N', 'P', 'S', 'R', 'U', 'T', 'W', 'V',
       'Y', '[', 'Z', ']', '_', 'a', 'c', 'b', 'e', 'd', 'g', 'f', 'i',
       'h', 'm', 'l', 'o', 'n', 's', 'r', 'u', 't', 'y']

    def __init__(self, config):
        super(MolOneHotFeaturizer, self).__init__()
        self.max_len = config["max_len"]
        self.enc = OneHotEncoder().fit(np.array(self.smiles_char).reshape(-1, 1))

    def __call__(self, data):
        temp = [c if c in self.smiles_char else '?' for c in data]
        if len(temp) < self.max_len:
            temp = temp + ['?'] * (self.max_len - len(temp))
        else:
            temp = temp[:self.max_len]
        return torch.tensor(self.enc.transform(np.array(temp).reshape(-1, 1)).toarray().T)

class MolGraphFeaturizer(MoleculeFeaturizer):
    allowable_features = {
        'possible_atomic_num_list':       list(range(1, 119)) + ['misc'],
        'possible_formal_charge_list':    [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 'misc'],
        'possible_chirality_list':        [
            Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
            Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
            Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
            Chem.rdchem.ChiralType.CHI_OTHER
        ],
        'possible_hybridization_list':    [
            Chem.rdchem.HybridizationType.SP,
            Chem.rdchem.HybridizationType.SP2,
            Chem.rdchem.HybridizationType.SP3,
            Chem.rdchem.HybridizationType.SP3D,
            Chem.rdchem.HybridizationType.SP3D2,
            Chem.rdchem.HybridizationType.UNSPECIFIED,
            'misc'
        ],
        'possible_numH_list':             [0, 1, 2, 3, 4, 5, 6, 7, 8, 'misc'],
        'possible_implicit_valence_list': [0, 1, 2, 3, 4, 5, 6],
        'possible_degree_list':           [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 'misc'],
        'possible_number_radical_e_list': [0, 1, 2, 3, 4, 'misc'],
        'possible_is_aromatic_list':      [False, True],
        'possible_is_in_ring_list':       [False, True],
        'possible_bond_type_list':        [
            Chem.rdchem.BondType.SINGLE,
            Chem.rdchem.BondType.DOUBLE,
            Chem.rdchem.BondType.TRIPLE,
            Chem.rdchem.BondType.AROMATIC,
            'misc'
        ],
        'possible_bond_dirs':             [  # only for double bond stereo information
            Chem.rdchem.BondDir.NONE,
            Chem.rdchem.BondDir.ENDUPRIGHT,
            Chem.rdchem.BondDir.ENDDOWNRIGHT
        ],
        'possible_bond_stereo_list':      [
            Chem.rdchem.BondStereo.STEREONONE,
            Chem.rdchem.BondStereo.STEREOZ,
            Chem.rdchem.BondStereo.STEREOE,
            Chem.rdchem.BondStereo.STEREOCIS,
            Chem.rdchem.BondStereo.STEREOTRANS,
            Chem.rdchem.BondStereo.STEREOANY,
        ], 
        'possible_is_conjugated_list':    [False, True]
    }

    def __init__(self, config):
        super(MolGraphFeaturizer, self).__init__()
        self.config = config
        if self.config["name"] == "unimap":
            self.allowable_features["possible_atomic_num_list"] = self.allowable_features["possible_atomic_num_list"][:-1] + ['[MASK]', 'misc']
            self.allowable_features["possible_bond_type_list"] = self.allowable_features["possible_bond_type_list"][:-1] + ['[MASK]', '[SELF]', 'misc']
            self.allowable_features["possible_bond_stereo_list"] = self.allowable_features["possible_bond_stereo_list"] + ['[MASK]']
            self.allowable_features["possible_hybridization_list"] = self.allowable_features["possible_hybridization_list"][:-2] + ['misc']

    def __call__(self, data):
        if isinstance(data, Molecule):
            mol = Chem.MolFromSmiles(data.smiles)
            # mol = AllChem.MolFromSmiles(data)
        else:
            mol = data
        # atoms
        atom_features_list = []
        for atom in mol.GetAtoms():
            if self.config["name"] in ["ogb", "unimap"]:
                atom_feature = [
                    safe_index(self.allowable_features['possible_atomic_num_list'], atom.GetAtomicNum()),
                    self.allowable_features['possible_chirality_list'].index(atom.GetChiralTag()),
                    safe_index(self.allowable_features['possible_degree_list'], atom.GetTotalDegree()),
                    safe_index(self.allowable_features['possible_formal_charge_list'], atom.GetFormalCharge()),
                    safe_index(self.allowable_features['possible_numH_list'], atom.GetTotalNumHs()),
                    safe_index(self.allowable_features['possible_number_radical_e_list'], atom.GetNumRadicalElectrons()),
                    safe_index(self.allowable_features['possible_hybridization_list'], atom.GetHybridization()),
                    self.allowable_features['possible_is_aromatic_list'].index(atom.GetIsAromatic()),
                    self.allowable_features['possible_is_in_ring_list'].index(atom.IsInRing()),
                ]
            else:
                atom_feature = [
                    safe_index(self.allowable_features['possible_atomic_num_list'], atom.GetAtomicNum()),
                    self.allowable_features['possible_chirality_list'].index(atom.GetChiralTag())
                ]
            atom_features_list.append(atom_feature)
        x = torch.tensor(np.array(atom_features_list), dtype=torch.long)

        # bonds
        if len(mol.GetBonds()) <= 0:  # mol has no bonds
            num_bond_features = 3 if self.config["name"] in ["ogb", "unimap"] else 2
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, num_bond_features), dtype=torch.long)
        else:  # mol has bonds
            edges_list = []
            edge_features_list = []
            for bond in mol.GetBonds():
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()
                if self.config["name"] in ["ogb", "unimap"]:
                    edge_feature = [
                        safe_index(self.allowable_features['possible_bond_type_list'], bond.GetBondType()),
                        self.allowable_features['possible_bond_stereo_list'].index(bond.GetStereo()),
                        self.allowable_features['possible_is_conjugated_list'].index(bond.GetIsConjugated()),
                    ]
                else:
                    edge_feature = [
                        self.allowable_features['possible_bond_type_list'].index(bond.GetBondType()),
                        self.allowable_features['possible_bond_dirs'].index(bond.GetBondDir())
                    ]
                edges_list.append((i, j))
                edge_features_list.append(edge_feature)
                edges_list.append((j, i))
                edge_features_list.append(edge_feature)

            # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
            edge_index = torch.tensor(np.array(edges_list).T, dtype=torch.long)

            # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
            edge_attr = torch.tensor(np.array(edge_features_list), dtype=torch.long)

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

        return data

class MolGGNNFeaturizer(MoleculeFeaturizer):
    def __init__(self, config):
        super(MolGGNNFeaturizer, self).__init__()
        self.max_n_atoms = config["max_n_atoms"]
        self.atomic_num_list = config["atomic_num_list"]
        self.bond_type_list = [
            Chem.rdchem.BondType.SINGLE,
            Chem.rdchem.BondType.DOUBLE,
            Chem.rdchem.BondType.TRIPLE,
            'misc'
        ]

    def __call__(self, data):
        if isinstance(data, str):
            mol = Chem.MolFromSmiles(data)
        else:
            mol = data
        Chem.Kekulize(mol)
        x = self._construct_atomic_number_array(mol, self.max_n_atoms)
        adj = self._construct_adj_matrix(mol, self.max_n_atoms)
        return x, adj, self._rescale_adj(adj) 

    def _construct_atomic_number_array(self, mol, out_size=-1):
        """Returns atomic numbers of atoms consisting a molecule.

        Args:
            mol (rdkit.Chem.Mol): Input molecule.
            out_size (int): The size of returned array.
                If this option is negative, it does not take any effect.
                Otherwise, it must be larger than the number of atoms
                in the input molecules. In that case, the tail of
                the array is padded with zeros.

        Returns:
            torch.tensor: a tensor consisting of atomic numbers
                of atoms in the molecule.
        """

        atom_list = [a.GetAtomicNum() for a in mol.GetAtoms()]
        if len(atom_list) > self.max_n_atoms:
            atom_list =  atom_list[:self.max_n_atoms]

        if out_size < 0:
            result = torch.zeros(len(atom_list), len(self.atomic_num_list))
        else:
            result = torch.zeros(out_size, len(self.atomic_num_list))
        for i, atom in enumerate(atom_list):
            result[i, safe_index(self.atomic_num_list, atom)] = 1
        for i in range(len(atom_list), self.max_n_atoms):
            result[i, -1] = 1
        return result

    def _construct_adj_matrix(self, mol, out_size=-1, self_connection=True):
        """Returns the adjacent matrix of the given molecule.

        This function returns the adjacent matrix of the given molecule.
        Contrary to the specification of
        :func:`rdkit.Chem.rdmolops.GetAdjacencyMatrix`,
        The diagonal entries of the returned matrix are all-one.

        Args:
            mol (rdkit.Chem.Mol): Input molecule.
            out_size (int): The size of the returned matrix.
                If this option is negative, it does not take any effect.
                Otherwise, it must be larger than the number of atoms
                in the input molecules. In that case, the adjacent
                matrix is expanded and zeros are padded to right
                columns and bottom rows.
            self_connection (bool): Add self connection or not.
                If True, diagonal element of adjacency matrix is filled with 1.

        Returns:
            adj (torch.tensor): The adjacent matrix of the input molecule.
                It is 2-dimensional tensor with shape (atoms1, atoms2), where
                atoms1 & atoms2 represent from and to of the edge respectively.
                If ``out_size`` is non-negative, the returned
                its size is equal to that value. Otherwise,
                it is equal to the number of atoms in the the molecule.
        """

        if out_size < 0:
            adj = torch.zeros(4, mol.GetNumAtoms(), mol.GetNumAtoms())
        else:
            adj = torch.zeros(4, out_size, out_size)
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            adj[safe_index(self.bond_type_list, bond.GetBondType()), i, j] = 1
            adj[safe_index(self.bond_type_list, bond.GetBondType()), j, i] = 1
        adj[3] = 1 - torch.sum(adj[:3], dim=0)
        return adj

    def _rescale_adj(self, adj):
        # Previous paper didn't use rescale_adj.
        # In their implementation, the normalization sum is: num_neighbors = F.sum(adj, axis=(1, 2))
        # In this implementation, the normaliztion term is different
        # raise NotImplementedError
        # (256,4,9, 9):
        # 4: single, double, triple, and bond between disconnected atoms (negative mask of sum of previous)
        # 1-adj[i,:3,:,:].sum(dim=0) == adj[i,4,:,:]
        # usually first 3 matrices have no diagnal, the last has.
        # A_prime = self.A + sp.eye(self.A.shape[0])
        num_neighbors = adj.sum(dim=(0, 1)).float()
        num_neighbors_inv = num_neighbors.pow(-1)
        num_neighbors_inv[num_neighbors_inv == float('inf')] = 0
        adj_prime = num_neighbors_inv[None, None, :] * adj
        return adj_prime

class MolMGNNFeaturizer(MoleculeFeaturizer):
    allowable_atom_list = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na','Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb','Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H','Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr','Cr', 'Pt', 'Hg', 'Pb', 'Unknown']
    allowable_degree_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    allowable_num_hs_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    allowable_implicit_valence_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    allowable_hybridization_list = [
        Chem.rdchem.HybridizationType.SP, 
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3, 
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2, 
        'other'
    ]
    allowable_cip_code_list = ['R', 'S']

    def __init__(self, config):
        super(MolMGNNFeaturizer, self).__init__()
        self.config = config

    def __call__(self, data):
        if isinstance(data, str):
            mol = Chem.MolFromSmiles(data)
        else:
            mol = data
        
        atom_features_list = []
        for atom in mol.GetAtoms():
            encoding = self.one_of_k_encoding_unk(atom.GetSymbol(), self.allowable_atom_list)
            encoding += self.one_of_k_encoding(atom.GetDegree(), self.allowable_degree_list)
            encoding += self.one_of_k_encoding_unk(atom.GetTotalNumHs(), self.allowable_num_hs_list)
            encoding += self.one_of_k_encoding_unk(atom.GetImplicitValence(), self.allowable_implicit_valence_list)
            encoding += self.one_of_k_encoding_unk(atom.GetHybridization(), self.allowable_hybridization_list)
            encoding += [atom.GetIsAromatic()]
            try:
                encoding += self.one_of_k_encoding_unk(atom.GetProp("_CIPNode"), self.allowable_cip_code_list)
            except:
                encoding += [0, 0]
            encoding += [atom.HasProp("_ChiralityPossible")]
            encoding /= np.sum(encoding)
            atom_features_list.append(encoding)
        x = torch.tensor(np.array(atom_features_list), dtype=torch.float)

        if len(mol.GetBonds()) <= 0:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        else:
            edges_list = []
            for bond in mol.GetBonds():
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()
                edges_list.append((i, j))
                edges_list.append((j, i))
            edge_index = torch.tensor(np.array(edges_list).T, dtype=torch.long)
        
        return Data(x=x, edge_index=edge_index)

    def one_of_k_encoding(self, x, allowable_set):
        if x not in allowable_set:
            raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
        return list(map(lambda s: x == s, allowable_set))

    def one_of_k_encoding_unk(self, x, allowable_set):
        """Maps inputs not in the allowable set to the last element."""
        if x not in allowable_set:
            x = allowable_set[-1]
        return list(map(lambda s: x == s, allowable_set))
