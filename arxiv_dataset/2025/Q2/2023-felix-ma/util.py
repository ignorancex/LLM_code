from matplotlib import pyplot as plt
import matplotlib
import torch
import pandas
import networkx as nx
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
from toolz.itertoolz import partition
from toolz.dicttoolz import keyfilter


def plot_embeddings(embeddings, labels, name="embeddings.pdf"):
    embeddings = embeddings.detach().cpu()
    embeddings = TSNE(n_components=2).fit_transform(embeddings)
    xs = embeddings[:,0]
    ys = embeddings[:,1]
    if labels is None:
        labels = torch.tensor([0] * len(xs))
    plt.scatter(xs, ys, c=labels.detach().cpu())
    plt.savefig(name)
    

def plot_clusterer(clusterer, dataset):
    labels = clusterer.predict(dataset.x)
    plot_embeddings(dataset.x, labels)
    

def plot_confusion_matrix(xs,ys):
    matrix = confusion_matrix(xs,ys)
    plt.imshow(matrix, cmap='hot', interpolation='nearest')

def save_results(results, file_name):
    df = pandas.DataFrame()
    

def cond(value, *clauses):
    """
    similar to https://clojuredocs.org/clojure.core/case
    :param value: Value to be matches against the clauses
    :param clauses: Clauses to be matched
    :return: result of matched expression
    """
    clauses = partition(2, clauses, pad=None)
    for clause in clauses:
        test, result = clause
        if result is None: # default case
            return test
        elif type(test) is tuple:
            for t in test:
                if t == value:
                    return result
        elif test == value:
            return result
    return None
        
def select_keys(dictionary, keys):
    """
    similar to https://clojuredocs.org/clojure.core/select-keys
    ignores keys not present in dictionary
    :return: new dictionary with only given keys
    """
    return keyfilter(lambda key: key in keys, dictionary)
