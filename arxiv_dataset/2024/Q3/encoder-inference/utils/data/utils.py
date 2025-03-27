from typing import Callable
import os, torch, numpy as np


if os.getenv("TORCH_HOME") is not None:
    TORCH_DATA_PATH = os.path.join(os.getenv("TORCH_HOME"), "data")
elif os.getenv("HOME") is not None:
    TORCH_DATA_PATH = os.path.join(os.getenv("HOME"), ".cache/torch/data")
else:
    TORCH_DATA_PATH = "./data"


class EmbedDataset:
    @staticmethod
    def build(dataset, encoder: Callable, cache_batch: int = 1000) -> "EmbeddingDataset":
        embed_x, y = [], []

        for i in range(0, len(dataset), cache_batch):
            ed = min(i + cache_batch, len(dataset))
            batch_x = []
            for k in range(i, ed):
                xx, yy = dataset[k]
                batch_x.append(xx)
                y.append(yy)
            embed_x.append(encoder(batch_x).cpu())

        embed_x = torch.cat(embed_x)
        y = torch.tensor(y)

        return EmbeddingDataset(embed_x, y, dataset)

    def __init__(self, embed_x, y, dataset):
        self.embed_x = embed_x
        self.y = y
        self.dataset = dataset
        self.embed_dims = len(embed_x[0])

    def __getitem__(self, idx):
        return self.embed_x[idx], self.y[idx]

    def __len__(self):
        return len(self.embed_x)


class Loader():
    def __init__(self, dataset, batch_size, train=True, num_workers=4):
        self.loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=train, drop_last=train, num_workers=num_workers)
        self.iterator = None

    def __iter__(self):
        return iter(self.loader)

    def __len__(self):
        return len(self.loader)

    def __next__(self):
        if self.iterator is None:
            self.iterator = iter(self.loader)

        try:
            samples = next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.loader)
            samples = next(self.iterator)

        return samples


def get_pre_embed_dataset(path=None):
    data = np.load(path)
    return EmbedDataset(data["embeds"], data["labels"], None)
