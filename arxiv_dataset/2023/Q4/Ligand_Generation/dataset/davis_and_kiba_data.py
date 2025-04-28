import os
import torch
import pickle 
import torch
from torch.utils.data import Dataset
from tdc.generation import MolGen 
from tdc.chem_utils import MolConvert
from tdc.chem_utils.oracle.oracle import smiles_to_rdkit_mol, qed, penalized_logp, jnk3, zaleplon_mpo, Vina_smiles, Vina_3d
from torch_geometric.data import Batch, Data
from torch_geometric.utils.smiles import from_smiles
from ogb.utils.mol import smiles2graph
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

def collate_fn_with_label(batch):
    protein_graph = [batch[i][0] for i in range(len(batch))]
    seq_embs = torch.stack([batch[i][1] for i in range(len(batch))])
    ligand_graph = [batch[i][2] for i in range(len(batch))]
    ligand_mp = torch.stack([batch[i][3] for i in range(len(batch))])
    label = torch.tensor([batch[i][4] for i in range(len(batch))])
    return Batch.from_data_list(protein_graph), seq_embs, Batch.from_data_list(ligand_graph), ligand_mp, label

def collate_fn_without_mp(batch):
    protein_graph = [batch[i][0] for i in range(len(batch))]
    seq_embs = torch.stack([batch[i][1] for i in range(len(batch))])
    ligand_graph = [batch[i][2] for i in range(len(batch))]
    label = torch.tensor([batch[i][3] for i in range(len(batch))])
    return Batch.from_data_list(protein_graph), seq_embs, Batch.from_data_list(ligand_graph), label

class Davis_Kiba_Dataset(Dataset):
    def __init__(self, root_path, df):
        super().__init__() 

        self.df = df
        self.root_path = root_path
        with open(os.path.join(self.root_path, "refine_esm_embedding.pkl"), "rb") as file:
            self.protein_to_esm = pickle.load(file)
        
        with open(os.path.join(self.root_path, "ligand_to_ecfp.pkl"), "rb") as f:
            self.ligand_mps= pickle.load(f)
        
        with open(os.path.join(self.root_path, "ligand_to_graph.pkl"), "rb") as f:
            self.ligand_graphs = pickle.load(f)

    def __len__(self):
        return self.df.shape[0]
    
    def get_lig_graph(self, smiles):
        graph = smiles2graph(smiles)
        return Data(x = torch.tensor(graph['node_feat']), edge_index = torch.tensor(graph['edge_index']), edge_attr = torch.tensor(graph['edge_feat']))

    def __getitem__(self, idx):
        item = self.df.iloc[idx]
        ligand_graph = self.get_lig_graph(item['ligand'])
        tensor_path = os.path.join(self.root_path, f"res_graph/{item['protein']}.pdb.pt")
        protein_graph = torch.load(tensor_path, map_location = "cpu")
        label = float(item['label'])
        esm_embedding = self.protein_to_esm[item['protein']]
        ligand_mp = torch.from_numpy(self.ligand_mps[item['ligand']])
        return protein_graph, esm_embedding, ligand_graph, ligand_mp, label
        
if __name__=="__main__":
    import pandas as pd
    csv_file = "/cm/shared/khangnn4/data/lba/data/davis/full.csv"
    root_path = "/cm/shared/khangnn4/data/lba/data/davis"
    df = pd.read_csv(csv_file)
    dataset = DavisDataset(root_path, df)
    protein_graph, esm_embedding, ligand_graph, ligand_mp, label = dataset[0]
    print(ligand_graph)