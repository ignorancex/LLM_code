import pickle 
import torch
from torch.utils.data import Dataset
from tdc.chem_utils import MolConvert
from torch_geometric.data import Batch, Data
from ogb.utils.mol import smiles2graph
import time
converter = MolConvert(src = 'SMILES', dst = 'SELFIES')
inverter = MolConvert(src='SELFIES', dst = 'SMILES')

splits = {
    "train" : "./data/pdbbind/timesplit_no_lig_or_rec_overlap_train", 
    "valid" : "./data/pdbbind/timesplit_no_lig_or_rec_overlap_val", 
    "test" : "./data/pdbbind/timesplit_test", 
    "train_label" : "./data/pdbbind/timesplit_no_lig_overlap_train", 
    "valid_label" : "./data/pdbbind/timesplit_no_lig_overlap_val"
}

print("loading all data")
with open("complex_2_smiles.pkl", "rb") as f:
    smiles_dict = pickle.load(f)

with open("invalid.pkl", "rb") as f:
    invalid = pickle.load(f)
    invalid_set = list(invalid.keys())

target_dict = torch.load("./data/pdbbind/all_graph_processed.pt")
print("Done Loading All Data !!")
save_model = './checkpoint/selfies_vae_model_020.pt'
vae_model = torch.load(save_model)
vocabulary = vae_model.vocabulary

with open("./data/pdbbind/complex_2_esm.pkl", "rb") as f:
    complex_2_esm = pickle.load(f)

del vae_model


def load_pk_data(data_path):                                                                                                                                                                                       
    res = dict()                                                                                                                                                                                                          
    with open(data_path) as f:                                                                                                                                                                                            
        for line in f:                                                                                                                                                                                                    
            if '#' in line:                                                                                                                                                                                               
                continue                                                                                                                                                                                                  
            cont = line.strip().split()                                                                                                                                                                                   
            if len(cont) < 5:                                                                                                                                                                                             
                continue                                                                                                                                                                                                  
            code, pk = cont[0], cont[3]                                                                                                                                                                                   
            res[code] = float(pk)                                                                                                                                                                                         
    return res   

class PDBBindDataset(Dataset):
    def __init__(self, split, device ="cuda"):
        super().__init__()

        with open(splits[split], "r") as f:
            self.complex_list = f.readlines()
            self.complex_list = list(map(lambda x: x.strip(), self.complex_list))
            self.complex_list = list(filter(lambda x: x not in invalid_set, self.complex_list))
            
        self.device = device

    def __len__(self):
        return len(self.complex_list)
    
    def __getitem__(self, idx):
        complex_name = self.complex_list[idx]
        selfies_string = converter(smiles_dict[complex_name])
        input_ids = torch.tensor(vocabulary.string2ids(selfies_string)).to(self.device)
        seq_emb = complex_2_esm[complex_name]
        return target_dict[complex_name].to(self.device), input_ids.to(self.device), seq_emb.to(self.device)

class PDBBindDataset_with_Label(Dataset):
    def __init__(self, split, device = "cuda"):
        super().__init__()
        with open(splits[split], "r") as f:
            self.complex_list = f.readlines()
            self.complex_list = list(map(lambda x: x.strip(), self.complex_list)) 
        pk_path = "./data/pdbbind/index/INDEX_general_PL_data.2020"
        self.pk_dict = load_pk_data(pk_path)

    def __len__(self):
        return len(self.complex_list)

    def get_lig_graph(self, smiles):
        graph = smiles2graph(smiles)
        return Data(x = torch.tensor(graph['node_feat']), edge_index = torch.tensor(graph['edge_index']), edge_attr = torch.tensor(graph['edge_feat']))


    def __getitem__(self, idx):
        complex_name = self.complex_list[idx]
        protein_graph = target_dict[complex_name]
        seq_emb = complex_2_esm[complex_name]
        ligand_graph = self.get_lig_graph(smiles_dict[complex_name])
        label = self.pk_dict[complex_name]
        return protein_graph, seq_emb, ligand_graph, label

def collate_fn(batch):
    graph_list = [batch[i][0] for i in range(len(batch))]
    input_ids = [batch[i][1] for i in range(len(batch))]
    seq_embs = torch.stack([batch[i][2] for i in range(len(batch))])
    return Batch.from_data_list(graph_list), input_ids, seq_embs

def collate_fn_with_label(batch):
    protein_graph = [batch[i][0] for i in range(len(batch))]
    seq_embs = torch.stack([batch[i][1] for i in range(len(batch))])
    ligand_graph = [batch[i][2] for i in range(len(batch))]
    label = torch.tensor([batch[i][3] for i in range(len(batch))])
    return Batch.from_data_list(protein_graph), seq_embs, Batch.from_data_list(ligand_graph), label

if __name__ == "__main__":
    dataset = PDBBindDataset_with_Label("train")
    from torch.utils.data import DataLoader
    train_loader = DataLoader(dataset, batch_size = 3, collate_fn = collate_fn_with_label)
    for batch in train_loader:
        protein_graph, seq_emb, ligand_graph, label = batch
        print(protein_graph)
        print(seq_emb.shape)
        print(ligand_graph)
        print(label)
        exit(0)
    