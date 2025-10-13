from glob import glob
import json

import networkx as nx
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from gc_bert.utils import split


"""
https://paperswithcode.com/dataset/pubmed
"""

class TextGraphDataset(Dataset):

    def __init__(self, data_path, citation_path, transform=None, target_transform=None, return_idx=False, split_seed=1):
        self.path = data_path
        self.citation_path = citation_path
        self.transform = transform
        self.target_transform = target_transform
        self.return_idx = return_idx
        self.split_seed = split_seed
        self.df = None
        self.edges = None
        self.G = None
        self.adj = None
        self.mode = None
        self.mask = None
        self.add_gc_token = False
    
    def remove_disconnected_nodes(self):
        to_exclude = set(self.df.index) - set(self.edges.target.astype(int)) - set(self.edges.source.astype(int))
        print(f'Excluding {len(to_exclude)} documents.')
        self.df = self.df.loc[~self.df.index.isin(to_exclude)]
        idx_remap = dict(zip(self.df.index.tolist(), range(len(self.df))))
        self.edges = self.edges.assign(
            source=self.edges.source.map(idx_remap),
            target=self.edges.target.map(idx_remap)
        )
        self.df = self.df.reset_index(drop=True)

    def split_data(self):
        idx_train, idx_valid, idx_test = split(self.df.shape[0], self.split_seed)
        self.df.loc[idx_train, 'mode'] = 'train'
        self.df.loc[idx_valid, 'mode'] = 'valid'
        self.df.loc[idx_test, 'mode'] = 'test'

    def change_mode(self, mode):
        if mode in ['train', 'valid', 'test']:
            self.mode = mode
            self.mask = (self.df['mode'] == mode).values
        else:
            raise Exception(f'Inproper mode {mode}')

    @property
    def labels(self):
        if self.mode is None:
            return torch.tensor(self.df.label.tolist())
        else:
            return torch.tensor(
                self.df.loc[self.mask].label.tolist()
            )

    @property
    def num_labels(self):
        return len(self.df.label.unique())

    @property
    def real_len(self):
        return len(self.df)

    def __len__(self):
        if self.mode is None:
            return len(self.df)
        else:
            return len(self.df.loc[self.mask])

    def __getitem__(self, idx):
        real_idx, text, label = self.df.loc[
            self.mask, ['text', 'label']].reset_index().iloc[idx]
        if self.add_gc_token:
            text = '<GC> ' + text
        if self.transform:
            text = self.transform(text)
        if self.target_transform:
            label = self.target_transform(label)
        if self.return_idx:
            return text, label, real_idx
        else:
            return text, label

    def create_graph(self):
        self.G = nx.Graph()
        self.G.add_nodes_from(self.df.index.tolist())
        self.G.add_edges_from(self.edges.values.tolist())
        return self.G

    def create_adj_matrix(self, to_sparse=True):
        if self.G is None:
            self.create_graph()

        if to_sparse:
            self.adj = nx.convert_matrix.to_scipy_sparse_matrix(
                self.G, nodelist=self.df.index.tolist(), dtype='float32',
            )
        else:
            self.adj = nx.convert_matrix.to_numpy_array(
                self.G, nodelist=self.df.index.tolist()
            )
        self.edge_idx = torch.tensor(np.array(self.adj.nonzero()), dtype=torch.int64)
        
        return self.adj

class PubmedDataset(TextGraphDataset):

    def load_data(self):
        with open(self.path) as f:
            self.df = pd.DataFrame(json.load(f))
        self.edges = pd.read_csv(self.citation_path)
        self.clean_data()
        self.split_data()

    def clean_data(self):
        self.df = (
            self.df
            .loc[(self.df.label.notna()) & (~self.df.pmid.duplicated())
                & (self.df.abstract.notna())]
            .pipe(lambda df: df.assign(
                label=df.label.astype(int),
                text_id=df.pmid.astype(int),
                time=pd.to_datetime(self.df.history.apply(lambda x: min(x.values()))),
            ))
            .reset_index(drop=True)
            .rename(columns={'abstract': 'text'})
            [['text', 'label', 'text_id', 'authors', 'title', 'time']]
        )
        text_id_to_id = (
            self.df
            .reset_index()
            .set_index('text_id')
            ['index']
            .to_dict()
        )
        self.edges = (
            self.edges
            .dropna()
            .loc[(self.edges.source.isin(self.df.text_id)) &
                (self.edges.target.isin(self.df.text_id)) &
                (~self.edges.duplicated())]
            .assign(
                source=self.edges.source.map(text_id_to_id),
                target=self.edges.target.map(text_id_to_id),
            )
        )
        self.remove_disconnected_nodes()

    def create_authors_list(self):
        self.authors = set.union(*[set(i) for i in self.df.authors.tolist()])



class WikiDataset(TextGraphDataset):

    def clean_data(self):
        pass

    def load_edges(self):
        with open(self.citation_path, 'r') as f:
            data = json.load(f)
        links = []
        for article, art_links in data.items():
            links.extend([article, link] for link in art_links)

        links = pd.DataFrame(links, columns=['source', 'target'])
        title_to_id = self.df.set_index('title')['text_id'].to_dict()
        self.edges = (
            links
            .assign(
                source=links.source.map(title_to_id),
                target=links.target.map(title_to_id)
            )
        )

    def load_data(self):
        train = pd.read_csv(
            'dbpedia/train.csv', names=['label', 'title', 'text']
        ).assign(mode='train')
        test = pd.read_csv(
            'dbpedia/test.csv', names=['label', 'title', 'text']
        ).assign(mode='test')
        self.df = (
            pd.concat([train, test])
            .reset_index(drop=True)
            .reset_index()
            .rename(columns={'index': 'text_id'})
        )
        self.load_edges()
        # self.df = self.df.loc[
        #     self.df.text_id.isin(set(self.edges.source) | set(self.edges.target))
        # ].reset_index(drop=True)
        self.remove_disconnected_nodes()
