import numpy as np
import os
import pandas as pd
import torch

class MSPDataset(torch.utils.data.Dataset):
    def __init__(self, df, path, target_transform=None):
        self.df = df
        self.target_transform = target_transform
        self.path = path
    def __len__(self):
        return len(self.df)
    def __getitem__(self, item):
        file = self.df.loc[item, "FileName"]
        label = self.df.loc[item, "EmoClass"]
        if self.target_transform is not None:
            label = self.target_transform(label)
        text_file = os.path.join(self.path, file.replace(".wav", ".txt"))
        with open(text_file, "r") as fp:
            text = fp.read().lower()
        return text, label

    
class AIBODataset(torch.utils.data.Dataset):
    def __init__(self, df, transliteration, target_transform=None):
        self.df = df
        self.target_transform = target_transform
        self.transliteration = transliteration
        self.indices = self.df.index
    def __len__(self):
        return len(self.df)
    def __getitem__(self, item):
        index = self.indices[item]
        file = self.df.loc[index, "id"]
        label = self.df.loc[index, "class"]
        if self.target_transform is not None:
            label = self.target_transform(label)
        text = self.transliteration.loc[file, "text"]
        return text, label
