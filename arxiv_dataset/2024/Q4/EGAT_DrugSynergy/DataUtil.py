from torch_geometric.data import InMemoryDataset, DataLoader
import pandas as pd
import torch
from processor import Processor_drug,Processor_motif
from tqdm import tqdm
import numpy as np
import random


class Datasetdrug1(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [self.root + "/raw/DrugCombRaw.csv", self.root + "/raw/Gene_Expressions.csv"]

    @property
    def processed_file_names(self):
        return "DrugComb_Drug1.pt"

    def download(self):
        pass

    def process(self):
        dataset = pd.read_csv(self.root + "/raw/DrugCombRaw.csv")
        expressions = pd.read_csv(self.root + "/raw/Gene_Expressions.csv")

        dataprocessor = Processor_drug(dataset, expressions)
        data_list, _ = dataprocessor.Process_Dataset()

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


class Datasetdrug2(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [self.root + "/raw/DrugCombRaw.csv", self.root + "/raw/Gene_Expressions.csv"]

    @property
    def processed_file_names(self):
        return "DrugComb_Drug2.pt"

    def download(self):
        pass

    def process(self):
        dataset = pd.read_csv(self.root + "/raw/DrugCombRaw.csv")
        expressions = pd.read_csv(self.root + "/raw/Gene_Expressions.csv")

        dataprocessor = Processor_drug(dataset, expressions)
        _, data_list = dataprocessor.Process_Dataset()
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


class Motif1(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [self.root + "/raw/DrugCombRaw.csv", self.root + "/raw/Gene_Expressions.csv"]

    @property
    def processed_file_names(self):
        return "DrugComb_Motif1.pt"

    def download(self):
        pass

    def process(self):
        dataset = torch.load(self.root + "/processed/DrugComb_Drug1.pt")

        dataprocessor = Processor_motif(dataset)
        data_list = dataprocessor.Process_Dataset()
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


class Motif2(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [self.root + "/raw/DrugCombRaw.csv", self.root + "/raw/Gene_Expressions.csv"]

    @property
    def processed_file_names(self):
        return "DrugComb_Motif2.pt"

    def download(self):
        pass

    def process(self):
        dataset = torch.load(self.root + "/processed/DrugComb_Drug2.pt")

        dataprocessor = Processor_motif(dataset)
        data_list = dataprocessor.Process_Dataset()
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


class GetData:

    def __init__(self, dataset1, dataset2, dataset3, dataset4, split):
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        self.dataset3 = dataset3
        self.dataset4 = dataset4

        temp = list(zip(self.dataset1, self.dataset2, self.dataset3, self.dataset4))
        random.seed(0)
        random.shuffle(temp)
        self.dataset1, self.dataset2, self.dataset3, self.dataset4 = zip(*temp)

        split1 = round(len(self.dataset1) * (split / 5))
        split2 = round(len(self.dataset1) * ((split + 1) / 5))

        self.TrainDataset_Drug1 = self.dataset1[:split1] + self.dataset1[split2:]
        self.TrainDataset_Drug2 = self.dataset2[:split1] + self.dataset2[split2:]
        self.TrainDataset_Motif1 = self.dataset3[:split1] + self.dataset3[split2:]
        self.TrainDataset_Motif2 = self.dataset4[:split1] + self.dataset4[split2:]

        self.TestDataset_Drug1 = self.dataset1[split1:split2]
        self.TestDataset_Drug2 = self.dataset2[split1:split2]
        self.TestDataset_Motif1 = self.dataset3[split1:split2]
        self.TestDataset_Motif2 = self.dataset4[split1:split2]

    def data_loader(self, batch_size):
        return (DataLoader(self.TrainDataset_Drug1, batch_size, shuffle=False),
                DataLoader(self.TrainDataset_Drug2, batch_size, shuffle=False),
                DataLoader(self.TrainDataset_Motif1, batch_size, shuffle=False),
                DataLoader(self.TrainDataset_Motif2, batch_size, shuffle=False)), \
            (DataLoader(self.TestDataset_Drug1, batch_size, shuffle=False),
             DataLoader(self.TestDataset_Drug2, batch_size, shuffle=False),
             DataLoader(self.TestDataset_Motif1, batch_size, shuffle=False),
             DataLoader(self.TestDataset_Motif2, batch_size, shuffle=False))