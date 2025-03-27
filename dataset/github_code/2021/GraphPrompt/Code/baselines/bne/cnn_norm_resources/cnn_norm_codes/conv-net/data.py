import torch
from torch.utils.data import Dataset
import os
import binaryfile
import sys
sys.path.insert(0, "/home/megh/projects/entity-norm/syn")
from bne_resources.run_bne_for_pt import do_tensorflow_routine

class RankingDataset(Dataset):
    def __init__(self, queries, mention2id, names, mode = 'train', emb_file = os.path.join("./vec_50.bin")):
        super().__init__()
        word2vec = open(emb_file,"rb")
        self.name_array = names
        self.mention2id =  mention2id
        self.query_id_array = queries
        self.name2id, self.query2id  = self.process_data_dict()
        self.write_all_name_to_file()
        current_path  = str(os.getcwd())
        batch_embeddings, embedding_name_dict = do_tensorflow_routine(os.path.join(current_path,"all_names_embeddings.txt"))

    def write_all_name_to_file(self):
        f = open("all_names_embeddings.txt","w")
        for idx, item in enumerate(self.mention2id):
            f.write(str(item) + "\n")

    def process_data_dict(self):
        name2id = {}
        for name_ in self.name_array:
            id_number =  self.mention2id[str(name_)]
            name2id[str(name_)] = id_number
        query2id  = {}
        for item in self.query_id_array:
            query2id[item] =  self.mention2id[item]
        return name2id, query2id

    def __len__(self):
        return len(self.query2id)

    def __getitem__(self, idx):
        

