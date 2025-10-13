"""Code for hyperparameter-tuning of thresholding and LGDE in newsgroups application"""

import pickle
import numpy as np
import pandas as pd

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm

from lgde import (
    Thresholding,
    KNearestNeighbors,
    IKEA,
    LGDEWithCDlib,
    LGDE,
    evaluate_prediction,
)

###################################
# data loading and pre-processing #
###################################

# load fine-tuned embeddings for different dimensions
embedding_50d = pd.read_pickle(
    "data/embeddings/glove_newsgroups_15_1.0_50d_5.0mu_df.pkl"
)

# retrieve word vectors at different dimensions
word_vecs_50d = np.zeros(
    (embedding_50d["word_vector"].shape[0], embedding_50d["word_vector"][0].shape[0])
)
for i in range(word_vecs_50d.shape[0]):
    word_vecs_50d[i, :] = embedding_50d["word_vector"][i]

# get word list
word_list = list(embedding_50d["word_string"])

# fetch train data
newsgroups_train = fetch_20newsgroups(
    subset="train", remove=("headers", "footers", "quotes")
)

# create binary split with 1: comp, misc and sci and 0: other groups
target_labels = np.asarray([1, 2, 3, 4, 5, 6, 11, 12, 13, 14])

# define training set
X_train = pd.Series(newsgroups_train["data"])
y_train = np.asarray(np.in1d(newsgroups_train["target"], target_labels), dtype=int)

# compute document-term matrix for train
vectorizer = CountVectorizer(vocabulary=word_list)
word_counts_train = vectorizer.fit_transform(X_train)
dt_train = pd.DataFrame(word_counts_train.toarray(), columns=word_list)

# seed dictionary
seed_dict = [
    "computer",
    "sys",
    "graphics",
    "os",
    "microsoft",
    "windows",
    "pc",
    "ibm",
    "hardware",
    "mac",
    "sale",
    "science",
    "cryptography",
    "electronics",
    "medicine",
    "space",
]

dimension_string = ["50", "100", "300"]
# mu_string = ["0.1", "0.5", "1.0", "1.5",
mu_string = ["5.0", "10.0"]


###############
# grid search #
###############

# iterate through all combinations of dim and mu
for dim_ind, dim in enumerate(dimension_string):

    # get word vectors
    embedding = pd.read_pickle(
        f"data/embeddings/glove_newsgroups_15_1.0_{dim}d_5.0mu_df.pkl"
    )
    word_vecs = np.zeros(
        (embedding["word_vector"].shape[0], embedding["word_vector"][0].shape[0])
    )
    for i in range(word_vecs.shape[0]):
        word_vecs[i, :] = embedding["word_vector"][i]

    ###############################################
    # grid search for thresholding hyperparameter #
    ###############################################

    epsilons = np.arange(0.3, 1, 0.001)
    size_disc_eps = np.zeros_like(epsilons)
    precision_eps = np.zeros_like(epsilons)
    recall_eps = np.zeros_like(epsilons)
    fscore_eps = np.zeros_like(epsilons)

    # create Thresholding object
    thres = Thresholding(seed_dict, word_list, word_vecs)

    for i, epsilon in tqdm(enumerate(epsilons), total=len(epsilons)):
        # compute epsilon balls around keywords
        thres.expand(epsilon=epsilon)
        size_disc_eps[i] = len(thres.discovered_dict_)
        # evaluate expanded dictionary on training set
        p, r, f = evaluate_prediction(
            thres.expanded_dict_, y_train, dt_train, with_report=False
        )
        precision_eps[i] = p
        recall_eps[i] = r
        fscore_eps[i] = f

    results_th = {
        "size": size_disc_eps,
        "precision": precision_eps,
        "recall": recall_eps,
        "fscore": fscore_eps,
        "eps": epsilons,
    }

    #######################################
    # grid search for kNN hyperparameters #
    #######################################

    ks = np.arange(1, 50)
    size_disc_knn = np.zeros(len(ks))
    precision_knn = np.zeros(len(ks))
    recall_knn = np.zeros(len(ks))
    fscore_knn = np.zeros(len(ks))

    # create kNN object
    knn = KNearestNeighbors(seed_dict, word_list, word_vecs)

    for i, k in tqdm(enumerate(ks), total=len(ks)):
        # compute k-most similar words of keywords
        knn.expand(k=k)
        size_disc_knn[i] = len(knn.discovered_dict_)
        # evaluate expanded dictionary on training set
        p, r, f = evaluate_prediction(
            knn.expanded_dict_, y_train, dt_train, with_report=False
        )
        precision_knn[i] = p
        recall_knn[i] = r
        fscore_knn[i] = f

    results_knn = {
        "size": size_disc_knn,
        "precision": precision_knn,
        "recall": recall_knn,
        "fscore": fscore_knn,
        "ks": ks,
    }

    #######################################
    # grid search for IKEA hyperparameter #
    #######################################

    epsilons = np.arange(0.3, 1, 0.001)
    size_disc_ikea = np.zeros_like(epsilons)
    precision_ikea = np.zeros_like(epsilons)
    recall_ikea = np.zeros_like(epsilons)
    fscore_ikea = np.zeros_like(epsilons)

    # create IKEA object
    ikea = IKEA(seed_dict, word_list, word_vecs)

    for i, epsilon in tqdm(enumerate(epsilons), total=len(epsilons)):
        # compute epsilon balls around keywords
        ikea.expand(epsilon=epsilon)
        size_disc_ikea[i] = len(ikea.discovered_dict_)
        # evaluate expanded dictionary on training set
        p, r, f = evaluate_prediction(
            ikea.expanded_dict_, y_train, dt_train, with_report=False
        )
        precision_ikea[i] = p
        recall_ikea[i] = r
        fscore_ikea[i] = f

    results_ikea = {
        "size": size_disc_ikea,
        "precision": precision_ikea,
        "recall": recall_ikea,
        "fscore": fscore_ikea,
        "eps": epsilons,
    }

    ##################################################
    # grid search for LGDE with LSWL hyperparameters #
    ##################################################

    ks_lswl = np.arange(3, 21, dtype=int)
    size_disc_lswl = np.zeros(len(ks_lswl))
    precision_lswl = np.zeros(len(ks_lswl))
    recall_lswl = np.zeros(len(ks_lswl))
    fscore_lswl = np.zeros(len(ks_lswl))
    communities_lswl = []

    # create LGDEWithCDlib object
    lswl = LGDEWithCDlib(seed_dict, word_list, word_vecs)

    for i, k in tqdm(enumerate(ks_lswl), total=len(ks_lswl)):
        # expand with LGDE based on LSWL
        lswl.expand(k=k, method="lswl", disable_tqdm=True)
        size_disc_lswl[i] = len(lswl.discovered_dict_)
        communities_lswl.append(lswl.semantic_communities_)
        # evaluate expanded dictionary on training set
        p, r, f = evaluate_prediction(
            lswl.expanded_dict_, y_train, dt_train, with_report=False
        )
        precision_lswl[i] = p
        recall_lswl[i] = r
        fscore_lswl[i] = f

    # store results
    results_lswl = {
        "size": size_disc_lswl,
        "precision": precision_lswl,
        "recall": recall_lswl,
        "fscore": fscore_lswl,
        "communities": communities_lswl,
        "ks": ks_lswl,
    }

    ###############################################
    # grid search for LGDE / SIWO hyperparameters #
    ###############################################

    times = np.arange(1, 11, dtype=int)
    ks = np.arange(3, 21, dtype=int)
    size_disc = np.zeros((len(ks), len(times)))
    precision = np.zeros((len(ks), len(times)))
    recall = np.zeros((len(ks), len(times)))
    fscore = np.zeros((len(ks), len(times)))
    communities = []

    # create LGDE object
    lgde = LGDE(seed_dict, word_list, word_vecs)

    # perform grid search
    for i, k in tqdm(enumerate(ks), total=len(ks)):
        lgde.construct_network(k=k)

        for j, t in tqdm(enumerate(times), total=len(times)):
            lgde.detect_local_communities(t=t, disable_tqdm=True)
            # store communities and compute size of discovered dictionary
            communities.append(lgde.semantic_communities_)
            size_disc[i, j] = len(lgde.discovered_dict_)
            # evaluate expanded dictionary on training set
            p, r, f = evaluate_prediction(
                lgde.expanded_dict_, y_train, dt_train, with_report=False
            )
            precision[i, j] = p
            recall[i, j] = r
            fscore[i, j] = f

    # store results
    results_lgde = {
        "size": size_disc,
        "precision": precision,
        "recall": recall,
        "fscore": fscore,
        "communities": communities,
        "times": times,
        "ks": ks,
    }

    # store all results in dictionary of dictionaries
    results = {
        "th": results_th,
        "lgde": results_lgde,
        "knn": results_knn,
        "ikea": results_ikea,
        "lswl": results_lswl,
    }

    with open(
        f"data/gs/gs_baselines_glove_newsgroups_15_1.0_{dim}d_5.0mu_df.pkl", "wb"
    ) as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
