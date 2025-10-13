#imported libraries
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_networkx
from torch_geometric.nn import LabelPropagation
from networkx import pagerank
from scipy.stats import entropy
from sklearn_extra.cluster import KMedoids
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score
import random
import copy
import matplotlib
import itertools
import pandas
import operator
import yaml
import numpy as np
import math
from datetime import datetime
from functools import partial
from itertools import takewhile
from toolz.itertoolz import iterate, first, concat, cons
from toolz.functoolz import thread_last, pipe
from toolz.dicttoolz import merge, valmap, keyfilter, get_in, merge_with, keymap
import matplotlib.pyplot as plt

#own libraries
import datasets
import sampling
from util import cond, plot_embeddings, plot_clusterer

#disable cuda
#torch.cuda.is_available = lambda: False

device = "cuda" if torch.cuda.is_available() else "cpu"

def accuracy(predictions, true_labels, mask):
    """Calculates accuracy, macro-f1 and the confusion matrix

    :param predictions: Predicted labels
    :param true_labels: Ground truth
    :param mask: Mask for instance selection
    :returns: Dict of accuracy scores (acc, macro-f1, and confusion matrix)
    """
    predictions = predictions.cpu()
    true_labels = true_labels.cpu()
    mask = mask.cpu()
    if mask.sum() == 0 or len(mask) == 0:
        return {"accuracy": 0,
                "macro-f1": 0,
                "confusion": [[]],
                "class accuracies": [],}
    else:
        return{"accuracy": accuracy_score(predictions[mask], true_labels[mask]),
               "macro-f1": f1_score(true_labels[mask], predictions[mask], average='macro'),
               "confusion": confusion_matrix(true_labels[mask], predictions[mask]).tolist(),
               "class accuracies": (confusion_matrix(true_labels[mask], predictions[mask]).diagonal()/confusion_matrix(true_labels[mask], predictions[mask]).sum(axis=0)).tolist()}

def few_shot_training(optimizer, model, dataset, report_only=False):
    """
    Trains and alters given model using few shot learning

    :param optimizer: Optimizer
    :param model: Model to train on
    :param dataset: Dataset with train/validation/test split
    :returns: Dictionary of train and test statistics
    """
    def get_train_validation_indices(dataset, validation_ratio=0.2):
        """
        Samples a sensible train/validation split for given dataset

        :param dataset: Dataset containing labels and masks
        :param validation_ratio: validation ratio per class
        :returns: indices for support/query/validation splits, support and query are further split by classes
        """
        train_prop_mask = torch.logical_or(dataset.train_mask, dataset.propagated_mask)
        vertices = torch.stack([dataset.y,torch.arange(len(dataset.y), device=device)]).T[train_prop_mask]
        buckets = {}
        training = []
        validation = []
        for label, index in vertices:
            label = label.item()
            index = index.item()
            if label in buckets:
                buckets[label].append(index)
            else:
                buckets[label] = [index]
        for label, indices in buckets.items():
            bucket_size = len(indices)
            # only add validation if enough samples are availible
            validation_size = math.ceil((bucket_size - 1) * validation_ratio)
            random.shuffle(indices)
            validation += indices[:validation_size]
            training += indices[validation_size:]
        return training, validation

    labels = None
    best_acc = 0
    no_increment_count = 0
    best_model_state = model.state_dict()
    training, validation = get_train_validation_indices(dataset)
    i = 0
    for _ in range(40 if len(validation) != 0 else 4): # train 4 epochs when validation set is empty
        i+=1
        accs = []
        acc = 0
        if(not report_only):
            model.train()
            optimizer.zero_grad()
            logits = model(dataset.x, dataset.edge_index, dataset.y, dataset.train_mask) # calculating logits updates the embeddings
            loss = model.loss(dataset, logits, training)
            loss.backward()
            optimizer.step()
        model.eval()
        labels = model(dataset.x, dataset.edge_index, dataset.y, dataset.train_mask).argmax(dim=1)
        acc = 0 if(len(validation) == 0) else accuracy_score(labels[validation].cpu(), dataset.y[validation].cpu())
        if acc > best_acc:
            best_model_state = model.state_dict()
            best_acc = acc
            no_increment_count = 0
        else:
            if no_increment_count < 4:
                no_increment_count += 1
            else:
                break
    model.load_state_dict(best_model_state)
    return merge(keymap(lambda key: "Labeled " + key, accuracy(labels, dataset.y, dataset.train_mask)),
                 keymap(lambda key: "Unlabeled " + key, accuracy(labels, dataset.y, dataset.val_mask)),
                 keymap(lambda key: "Test " + key, accuracy(labels, dataset.y, dataset.test_mask)))

def label_propagation(model, dataset, steps=2, uncertainty_threshold=0.2):
    """
    Propagates trainig labels and adds their labels to y, modifies dataset
    """
    dataset.propagated_mask = torch.zeros_like(dataset.propagated_mask, dtype=torch.bool)
    dataset.y = dataset.ground_truth.clone() # for sanity
    propagator = LabelPropagation(num_layers=steps, alpha=0.9)
    logits = propagator(dataset.ground_truth, dataset.edge_index, mask=dataset.train_mask)
    labels = logits.argmax(dim=-1)
    propagated_logits = logits.nonzero(as_tuple=True)[0]
    # remove uncertain logits
    nonzero_indices = torch.nonzero(logits, as_tuple=True)[0]
    normalized_uncertainty_scores = torch.ones(len(dataset.propagated_mask),device=device)
    entrophies = entropy(logits[nonzero_indices].T.cpu()) / math.log(dataset.num_classes)
    normalized_uncertainty_scores[nonzero_indices] = (torch.from_numpy(entrophies) / math.log(dataset.num_classes)).to(device)
    valid_uncertainty_mask = normalized_uncertainty_scores < uncertainty_threshold
    dataset.propagated_mask = torch.logical_and(valid_uncertainty_mask, dataset.val_mask)
    dataset.propagated_mask[dataset.test_mask] = False # prevent test leak
    dataset.propagated_mask[dataset.train_mask] = False # already labeled vertices don't get a pseudolabel
    dataset.y[dataset.propagated_mask] = labels[dataset.propagated_mask]
    #print("Propagated", dataset.propagated_mask.sum(), "labels\twrong samples:", (dataset.y != dataset.ground_truth).sum(), "\t", uncertainty_threshold)

def run(model,
        dataset_name,
        sampler='own',
        runs=10,
        label_propagation_uncertainty_treshold=0.2,
        num_steps=10,
        samples_per_step=1,
        seed=133742069,
        learning_rate=0.001,
        subsampler = "random",
        corruption = 0):
    """
    Runs experiments on given model "runs" times with the same settings

    :param model: Model constructor with 0 args for model construction
    :param dataset_name: Dataset name for training, testing, and validation
    :param sampler: Sampler used for the active learner
    :param runs: Number of experiment repeats
    :param budget: Number of samples moving from unlabeled to labeled
    :param seed: Initial seed for each strategy. Seed will change each run
    :param train_epochs: Number of epochs to train between samling
    :param learning_rate: Learning rate of the optimizer
    :param subsamler: Subsamling strategy
    :param corruption: Relative amount of wrong labeled instances
    :returns: run statistics
    """
    def run_once(model, dataset):
        """
        Runs experiments on given model "runs" times with the same settings
        
        :param model: Constructed model
        :param dataset: Data for training, testing, and validation
        :param budget: Number of samples moving from unlabeled to labeled
        :param learning_rate: Learning rate of the optimizer
        :returns: single run statistics
        """
        start_time = datetime.now()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4)
        classifier = None
        sampler_fun = sampling.sampler[sampler]
        # report initial stats
        run_stats = [merge(few_shot_training(optimizer, model, dataset, report_only=True),
                           {"Budget used": 0,
                            "Class distrubution": []})]
        budget = num_steps * samples_per_step * dataset.orig_num_classes
        while(dataset.train_mask.sum() < budget):
            # ask active learner for vertices
            sampled_indices = sampler_fun(n=min(dataset.num_classes * samples_per_step, budget - dataset.train_mask.sum()),
                                          model=model,
                                          dataset=dataset,
                                          perfect=False,
                                          subsampler=subsampler)
            if len(sampled_indices) != dataset.num_classes * samples_per_step:
                print("Warning: didn't sample |C| vertices: ", len(sampled_indices), "/", dataset.num_classes * samples_per_step)
            # move sampled vertices from the validation to the training set, also restore propagated indices if applicable
            dataset.val_mask[sampled_indices] = False
            dataset.train_mask[sampled_indices] = True
            # apply label propagation
            label_propagation(model, dataset, uncertainty_threshold=label_propagation_uncertainty_treshold)
            dataset.y[dataset.train_mask] = dataset.ground_truth[dataset.train_mask]
            # perform training
            run_stats.append(
                merge(few_shot_training(optimizer, model, dataset),
                      {"Budget used": int(dataset.train_mask.sum()),
                       "Class distrubution": torch.bincount(dataset.ground_truth[dataset.train_mask]).cpu().numpy().tolist()}))
            # plot embeddings
            # embeddings = model.get_embeddings(dataset.x, dataset.edge_index)
            # prototypes = model.prototypes
            # plot_embeddings(torch.cat([embeddings,prototypes]).detach(),
                            #labels=torch.cat([dataset.ground_truth, torch.full([dataset.num_classes], -1, device=device)]))
            
        stop_time = datetime.now()
        print(stop_time - start_time, run_stats[-1]['Unlabeled accuracy'])
        return run_stats
    # perform experiments
    results = []
    random.seed(seed)
    seeds = [random.randrange(2**31) for i in range(runs)]
    for i in range(runs):
        torch.manual_seed(seeds[i]) # update seeds
        np.random.seed(seeds[i])
        random.seed(seeds[i])
        # reset dataset
        dataset = datasets.get_dataset(dataset_name, seed=seeds[i], corruption=corruption)
        model_instance = model()
        model_instance = model_instance.to(device)
        result = run_once(model_instance, copy.deepcopy(dataset))
        result = list(map(partial(merge, {"seed": seeds[i]}), result))
        result.append({}) # empty row marks end of run
        results += result
    return results


"""
dataset = datasets.get_dataset('Cora')
model = models.GPN(dataset)

# prototypical
rank = torch.tensor(list(pagerank(to_networkx(dataset)).values()))
gpn_model = partial(GPN,
                    num_node_features=dataset.num_node_features,
                    num_classes=dataset.num_classes,
                    pagerank_scores=rank,
                    embedding_dim=16,
                    dropout=0.5)
gcn_model = partial(GCN,
                    in_channels=dataset.num_node_features,
                    hidden_channels=128,
                    num_layers=2,
                    out_channels=dataset.num_classes,
                    dropout=0.5)

results_gpn = {}
results_gcn = {}
for sampler_name in sampling.sampler.keys():
    results_gpn[sampler_name] = run(model=gpn_model, dataset=dataset, sampler=sampler_name,runs=3, budget=100)
    continue
    results_gcn[sampler_name] = run(model=gcn_model, dataset=dataset, sampler=sampler_name,runs=5, budget=100)
    print(sampler_name + "\tgpn f1: " + str(sum(map(lambda d: d['test']['macro-f1'], results_gpn[sampler_name])) / len(results_gpn[sampler_name])))
    print(sampler_name + "\tgcn f1: " + str(sum(map(lambda d: d['test']['macro-f1'], results_gcn[sampler_name])) / len(results_gcn[sampler_name])))

result = results_gpn[sampler_name]
result

#embeddings = result_gcn[0]['model'](dataset.x, dataset.edge_index)
model = results_gpn['random'][0]['model']
modified_dataset = results_gpn['random'][0]['dataset']
embeddings = model.embeddings
prototypes = model.prototypes
probabilities = model(dataset.x, dataset.edge_index).detach()
num_classes = dataset.y.unique().size(0)
plot_embeddings(torch.cat([embeddings,prototypes]).detach(),
                labels=torch.cat([dataset.y, torch.full([num_classes], num_classes)]))
plot_embeddings(dataset.x, dataset.y)
"""
