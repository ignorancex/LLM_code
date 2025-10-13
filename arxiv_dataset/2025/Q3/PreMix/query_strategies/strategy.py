import torch
import torch.nn as nn

from typing import Optional, Callable
from functools import partial

from source.training import Training
from source.dataset_utils import collate_features

class Strategy:
    def __init__(
        self,
        model: nn.Module,
        pool_dataset: torch.utils.data.Dataset,
        train_dataset: torch.utils.data.Dataset,
        collate_fn: Callable = partial(collate_features, label_type="int"),
        batch_size: Optional[int] = 1,
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.pool_dataset = pool_dataset
        self.collate_fn = collate_fn
        self.batch_size = batch_size

    def query(self, n):
        pass
    
    def predict(self, model, pool_dataset, collate_fn, batch_size):
        return Training.predict(model, pool_dataset, collate_fn, batch_size)

    def predict_prob(self, model, pool_dataset, collate_fn, batch_size):
        return Training.predict_prob(model, pool_dataset, collate_fn, batch_size)

    def predict_prob_embed(self, model, pool_dataset, collate_fn, batch_size):
        return Training.predict_prob_embed(model, pool_dataset, collate_fn, batch_size)

    def predict_all_representations(self, model, pool_dataset, collate_fn, batch_size):
       return Training.predict_all_representations(model, pool_dataset, collate_fn, batch_size)

    def predict_prob_dropout(self, model, pool_dataset, collate_fn, batch_size, n_drop):
        return Training.predict_prob_dropout(model, pool_dataset, collate_fn, batch_size, n_drop)

    def predict_prob_dropout_split(self, model, pool_dataset, collate_fn, batch_size, n_drop):
        return Training.predict_prob_dropout_split(model, pool_dataset, collate_fn, batch_size, n_drop)

    def predict_prob_embed_dropout_split(self, model, pool_dataset, collate_fn, batch_size, n_drop):
        return Training.predict_prob_embed_dropout_split(model, pool_dataset, collate_fn, batch_size, n_drop)

    def get_embedding(self, model, pool_dataset, collate_fn, batch_size):
        return Training.get_embedding(model, pool_dataset, collate_fn, batch_size)
    
    def get_embedding_train(self, model, train_dataset, collate_fn, batch_size):
        return Training.get_embedding(model, train_dataset, collate_fn, batch_size)

    def get_grad_embedding(self, model, pool_dataset, collate_fn, batch_size):
        return Training.get_grad_embedding(model, pool_dataset, collate_fn, batch_size)
