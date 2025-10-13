import torch

from torch_geometric.data import Dataset

class SynthDataset(Dataset):
    def __init__(self, dataset_name):
        super().__init__(dataset_name)
        self.dataset_name = dataset_name

    def get(self):
        data = torch.load(f"dataset/{self.dataset_name}Dataset/{self.dataset_name}Dataset.pt")
        return data

    def get_edge_split(self):
        split_edge = torch.load(f"dataset/{self.dataset_name}Dataset/{self.dataset_name}Dataset_split.pt")
        return split_edge

    def len(self):
        return 0