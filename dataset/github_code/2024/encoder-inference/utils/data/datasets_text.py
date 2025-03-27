from datasets import load_dataset

from .utils import EmbedDataset


class Yelp:
    def __init__(self, train=True):
        data = load_dataset("yelp_review_full")
        self.data = data["train" if train else "test"]
        self.train = train

    def __getitem__(self, idx):
        res = self.data[idx]
        return res["text"], res["label"]

    def __len__(self):
        return len(self.data)


class SST2:
    def __init__(self, train=True):
        data = load_dataset("sst2")
        self.data = data["train" if train else "validation"]
        self.train = train

    def __getitem__(self, idx):
        res = self.data[idx]
        return res["sentence"], res["label"]

    def __len__(self):
        return len(self.data)


class SST5:
    def __init__(self, train=True):
        data = load_dataset("sst", "default")
        self.data = data["train" if train else "validation"]
        self.train = train

    def __getitem__(self, idx):
        res = self.data[idx]
        text = res["sentence"]
        score = res["label"]

        if score <= 0.2: label = 0
        elif score <= 0.4: label = 1
        elif score <= 0.6: label = 2
        elif score <= 0.8: label = 3
        else: label = 4

        return text, label

    def __len__(self):
        return len(self.data)


class AgNews:
    def __init__(self, train=True):
        data = load_dataset("ag_news")
        self.data = data["train" if train else "test"]
        self.train = train

    def __getitem__(self, idx):
        res = self.data[idx]
        return res["text"], res["label"]

    def __len__(self):
        return len(self.data)


class TREC:
    def __init__(self, train=True):
        data = load_dataset("trec")
        self.data = data["train" if train else "test"]
        self.train = train

    def __getitem__(self, idx):
        res = self.data[idx]
        return res["text"], res["coarse_label"]

    def __len__(self):
        return len(self.data)


__dataset_zoo__ = {
    "yelp": Yelp,
    "sst2": SST2,
    "sst5": SST5,
    "agnews": AgNews,
    "trec": TREC,
}


def get_text_dataset(name: str, train=True):
    return __dataset_zoo__[name](train)


def get_text_embed_dataset(name: str, encoder, train=True, cache_batch: int = 1000):
    dataset = __dataset_zoo__[name](train)
    return EmbedDataset.build(dataset, encoder, cache_batch=cache_batch)
