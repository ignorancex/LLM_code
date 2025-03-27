from os import name
import torch
import sys
from net import *
from data import *
sys.path.insert(0, "/home/megh/projects/entity-norm/syn")
from code.dataset import *

class CNNNorm():
    def __init__(self,data_path = os.path.join("../../../data/datasets/cl.obo")) -> None:
        name_array, query_id_array, mention2id, edge_index  = load_data(filename=data_path)

        queries_train, queries_valid, queries_test = data_split(query_id_array = query_id_array, is_unseen=1, test_size=0.33)

        dummy_m = torch.rand(24,1,200,3)
        dummy_y = torch.rand(24,1,200,4)
        print("Dummy input sizes are:")
        print("m size ", dummy_m.shape)
        print("y size ", dummy_y.shape)

        self.dataset_ranking =  RankingDataset(queries_train, mention2id, name_array)
        model = RankingCNN()

        out  = model(dummy_m,dummy_y)
        
    def train():
        pass

if __name__ == "__main__":
    cnn_norm = CNNNorm()