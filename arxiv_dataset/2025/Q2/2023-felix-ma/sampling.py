import torch
import math
import json
import random
import numpy as np
from enum import Enum
from scipy.stats import entropy
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import f1_score
from torch.func import vmap
from torch.nn.functional import normalize
from toolz.functoolz import pipe, thread_first, identity, do
from toolz.itertoolz import groupby, first
from toolz.dicttoolz import valmap, dissoc
from functools import partial
from itertools import permutations
from torch_geometric.utils import degree
import torch.nn.functional as F

#disable cuda
#torch.cuda.is_available = lambda: False

device = "cuda" if torch.cuda.is_available() else "cpu"

def random_sampling(n, model, dataset, perfect=None, subsampler=None):
    """
    Selects vertices randomly.
    :param n: Number of samples to draw
    :param model: unused
    :param dataset: Data to sample from
    :param perfect: unused
    :param entopy_pagerank_weighting: unused
    :returns: Selected vertex indices
    """
    sampled_indices = torch.multinomial(dataset.val_mask.float(), n)
    return sampled_indices
    
    
def sub_sampler(num_samples, indices, embeddings, ranks, subsampler):
    """
    Selects vertices based on entropy or pagerank from selected indices.
    :param num_samples: Number of samples to draw
    :param indices: indices to sample from
    :param embeddings: for entropy calculation
    :param ranks: for pagerank weighting
    :param subsampler: subsampling strategy:random, pagerank, entropy, medoids
    :returns: Selected vertex indices
    """
    entropy_pagerank_weighting = {"random": -1,
                                  "pagerank": 0,
                                  "own": 0.5,
                                  "entropy": 1,
                                  "medoids": 2,}[subsampler]
    if entropy_pagerank_weighting <= 1:
        weights = torch.ones(len(indices))
        if(entropy_pagerank_weighting >= 0):
            normalized_entropies = pipe(embeddings[indices].T.cpu(),
                                        entropy,
                                        torch.from_numpy,
                                        partial(normalize, dim=0, p=1))
            normalized_pageranks = pipe(ranks[indices],
                                        partial(normalize, dim=0, p=1))
            weights = (normalized_pageranks * (1-entropy_pagerank_weighting)
                       + normalized_entropies * entropy_pagerank_weighting)
    
        normalized_weights = normalize(weights, dim=0, p=1).numpy().astype('float64')
        selected_indices = np.random.choice(indices,
                                            size=num_samples,
                                            p=(normalized_weights / normalized_weights.sum()), # normalize twice
                                            replace=False)
        return selected_indices
    else:
        clusterer = KMedoids(n_clusters=num_samples, init="k-medoids++").fit(embeddings[indices].cpu())
        selected_indices = np.array(indices)[clusterer.medoid_indices_]
        return selected_indices

def own_sampling(n, model, dataset, perfect=False, subsampler="random"):
    """
    Selects vertices of a dataset using the model as classifier to be included into the test set.
    :param n: Number of samples to draw
    :param model: model
    :param dataset: Data to sample from
    :param perfect: use oracle to get class assignments
    :param subsampler: subsampling strategy: random, pagerank, entropy, own, medoids
    :param compensate_undersampled: tries to sample more from undersampled classes
    :returns: Selected vertex indices
    """
    exclude_mask = torch.logical_or(dataset.train_mask,
                                    dataset.test_mask)
    # perform K-Medoids on the unlabeled embeddings
    clusterer = KMedoids(n_clusters=dataset.num_classes, init="k-medoids++").fit(model.get_embeddings(dataset.x, dataset.edge_index, dataset.ground_truth, dataset.train_mask)[dataset.val_mask].detach().cpu().numpy())
    labels = torch.full_like(dataset.ground_truth, -1) # initialize with -1 to ignore those vertices
    labels[dataset.val_mask] = torch.from_numpy(clusterer.labels_).to(device)
    if perfect: labels[dataset.val_mask] = dataset.ground_truth.clone()[dataset.val_mask]
    num_vertices = dataset.val_mask.sum()
    num_classes = dataset.num_classes
    samples_per_class = torch.zeros(dataset.num_classes)
    # one sample per class
    samples_per_class = torch.tensor([n // dataset.num_classes for i in range(dataset.num_classes)])
    remainder = n % dataset.num_classes
    if remainder > 0:
        samples_per_class[torch.multinomial(
            torch.ones(dataset.num_classes, dtype=float), remainder)] += 1
    # perform subsampling on each bucket
    sampled_indices = torch.tensor([], dtype=int)
    labels = labels.cpu().numpy()
    grouped_indices = groupby(lambda x: labels[x], range(0, len(dataset.y))) # group by label
    grouped_indices = dissoc(grouped_indices, -1) # remove train and test
    ranks = dataset.pagerank.cpu()
    for label, indices in grouped_indices.items():
        selected_indices = np.array([],dtype=int)
        if samples_per_class[label] > 0:
            selected_indices = sub_sampler(num_samples=int(min(samples_per_class[label], len(indices))),
                                           indices=indices,
                                           embeddings=model(dataset.x, dataset.edge_index, dataset.y, dataset.train_mask).detach(),
                                           ranks=ranks,
                                           subsampler=subsampler)
        sampled_indices = torch.cat([sampled_indices,
                                     torch.from_numpy(selected_indices)])
    return sampled_indices

def k_medoids_sampling(n, model, dataset, perfect=False, subsampler=None):    
    pass

sampler = {'random': random_sampling,
           'own': own_sampling,
           'k-medoids': k_medoids_sampling,}

"""
def classifier_sampling(n, model, dataset, labels):
    Selects vertices of a dataset using the labels from a classifier,
    which are used for increasing diversity.
    From those vertices the ones with the highest entropy and degree are sampled
    :param n: Number of samples to draw
    :param model: model
    :param dataset: Data to sample from
    :param labels: Labels to base the decision on
    :returns: Selected vertex indices
    exclude_mask = torch.logical_or(dataset.train_mask,
                                    dataset.test_mask)
    labels[exclude_mask] = dataset.num_classes # create an "excluded" class
    labels = labels.numpy() # convert to np for groupby
    # group indices by label and remove excluded indices
    grouped_indices = groupby(lambda x: labels[x], range(0, len(dataset.y)))
    grouped_indices = dissoc(grouped_indices, dataset.num_classes)
    #print(valmap(len, grouped_indices))
    # determine samples to be drawn per class
    logits = model(dataset.x, dataset.edge_index).detach()
    samples_per_class = torch.tensor([n // dataset.num_classes for i in range(dataset.num_classes)])
    remainder = n % dataset.num_classes
    if remainder > 0:
        samples_per_class[torch.multinomial(
            torch.ones(dataset.num_classes, dtype=float), remainder)] += 1
    # draw samples based on entropy and pagerank score
    sampled_indices = torch.tensor([], dtype=int)
    entopy_pagerank_weighting = 0
    for label, indices in grouped_indices.items():
        num_samples = int(min(samples_per_class[label], len(indices)))
        if(num_samples > 0):
            normalized_entropies = pipe(logits[indices].T,
                                        entropy,
                                        torch.from_numpy,
                                        partial(normalize, dim=0, p=1), 
                                        lambda e: torch.exp(-4 * torch.square(e - 1)),
                                        partial(normalize, dim=0, p=1))
            normalized_pageranks = pipe(dataset.pagerank[indices],
                                        partial(normalize, dim=0, p=1))
            weights = (normalized_pageranks * (1-entopy_pagerank_weighting)
                       + normalized_entropies * entopy_pagerank_weighting).numpy()
            #weights = np.ones_like(weights, dtype='float32')
            normalized_weights = weights / np.sum(weights)
            normalized_weights[-1] = 1 - np.sum(normalized_weights[0:-1])
            selected_indices = np.random.choice(indices,
                                                size=num_samples,
                                                p=normalized_weights,
                                                replace=False)
            sampled_indices = torch.cat([sampled_indices,
                                         torch.from_numpy(selected_indices)])
    return sampled_indices


sub_samplers = ['random', 'entropy', 'pagerank', 'own']

"""
