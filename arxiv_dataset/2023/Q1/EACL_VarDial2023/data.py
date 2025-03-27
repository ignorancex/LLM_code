import torch
from torch.utils.data import DataLoader
import pandas as pd


class VarDialDataset(torch.utils.data.Dataset):
    def __init__(self, df_path, label_mappings, num_classes, mode):
        super().__init__()
        self.df_path = df_path
        self.label_mappings = label_mappings
        self.num_classes = num_classes
        self.mode = mode
        if self.mode == "inference":
            self.df = pd.read_csv(
                self.df_path, sep="\t", names=["Text"], encoding="ISO-8859-1"
            )
        else:
            self.df = pd.read_csv(
                self.df_path,
                sep="\t",
                names=["Text", "Language"],
                encoding="ISO-8859-1",
            )
            if self.num_classes == 2:
                self.df.drop(self.df[self.df["Language"] == "EN"].index, inplace=True)
                self.df.drop(self.df[self.df["Language"] == "ES"].index, inplace=True)
                self.df.drop(self.df[self.df["Language"] == "PT"].index, inplace=True)
            self.df["Label"] = self.df.Language.map(self.label_mappings)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if self.mode == "inference":
            text = self.df["Text"].to_list()
            return text[idx], -1
        else:
            text = self.df["Text"].to_list()
            label = self.df["Label"].to_list()
            return text[idx], label[idx]

    def create_loader(self, batch_size):
        if self.mode == "train":
            weights = self.make_weights_for_balanced_classes()
            weights = torch.DoubleTensor(weights)
            sampler = torch.utils.data.sampler.WeightedRandomSampler(
                weights, len(weights)
            )
            return DataLoader(self, batch_size=batch_size, sampler=sampler)
        elif self.mode == "validation" or self.mode == "inference":
            return DataLoader(self, batch_size=batch_size, shuffle=False)

    def make_weights_for_balanced_classes(self):
        count = self.df["Label"].value_counts().to_list()
        class_weights = [1 / c for c in count]
        labels = self.df["Label"].to_list()
        sample_weights = [0] * len(labels)
        for idx, lbl in enumerate(labels):
            sample_weights[idx] = class_weights[lbl]
        return sample_weights
