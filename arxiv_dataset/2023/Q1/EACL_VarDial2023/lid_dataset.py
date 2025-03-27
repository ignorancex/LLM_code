import torch
from torch.utils.data import DataLoader


class LIDdataset(torch.utils.data.Dataset):
    def __init__(self, text, labels=None, return_labels=True):
        self.text = text
        self.return_labels = return_labels
        if self.return_labels:
            self.labels = labels

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        if self.return_labels:
            return self.text[idx], self.labels[idx]
        else:
            return self.text[idx]

    def create_loader(self, batch_size):
        return DataLoader(self, batch_size=batch_size, shuffle=False)
