import torch
import random
from torch.utils.data import Dataset


class EmbeddingDataset(Dataset):
    def __init__(self,
                 data,
                 rebalance=False,
                 device=None,
                 num_embeddings=-1):
        """ Embedding dataset

        :param data: embedding data for the dataset
        :param rebalance: whether to rebalance the data
        :param device: device to store data on
        :param num_embeddings: maximum number of embeddings
        """
        super().__init__()

        self.embeddings = torch.concatenate([torch.tensor(d['embeddings']).flatten(0,1) for d in data], dim=0)
        self.labels = torch.concatenate([torch.tensor(d['labels']).flatten(0,1) for d in data], dim=0)
        if num_embeddings > 0:
            indices = random.sample(list(range(0, self.embeddings.shape[0])), k=min(num_embeddings, self.embeddings.shape[0]))
            self.embeddings = self.embeddings[indices]
            self.labels = self.labels[indices]
        
        assert(self.embeddings.shape[0] == self.labels.shape[0])
        if rebalance:
            self._rebalance()
        if device is not None:
            self.embeddings = self.embeddings.to(device)
            self.labels = self.labels.to(device)
        
    def _rebalance(self):
        """ Rebalance the dataset """
        items_equality = (self.labels == 0)
        items_inequality = (self.labels != 0)
        num_eq, num_ineq = items_equality.sum(), items_inequality.sum()
        print("Rebalancing... {} eq and {} ineq".format(num_eq, num_ineq))
        if num_eq != num_ineq:
            if num_eq > num_ineq:
                emb = self.embeddings[items_inequality]
                lab = self.labels[items_inequality]
                num_sample = (num_eq-num_ineq)
            else: # num_eq < num_ineq:
                emb = self.embeddings[items_equality]
                lab = self.labels[items_equality]
                num_sample = (num_ineq-num_eq)
            
            subset_indices = random.choices(list(range(len(lab))), k=num_sample)
            self.embeddings = torch.concatenate([self.embeddings, emb[subset_indices]], dim=0)
            self.labels = torch.concatenate([self.labels, lab[subset_indices]], dim=0)
            print("New balance... {} eq and {} ineq".format(
                (self.labels == 0).sum(), (self.labels != 0).sum()
            ))
        else:
            print("Perfectly balanced... As all things should be")
            
    def get_all(self):
        """ Return all the data """
        return {
            'embeddings': self.embeddings,
            'labels': self.labels
        }
    
    def __len__(self):
        """ Get the length of the dataset """
        return self.embeddings.shape[0]
    
    def __getitem__(self, idx):
        """ Return a specific item in the dataset """
        return {
            'embeddings': self.embeddings[idx],
            'labels': self.labels[idx]
        }
