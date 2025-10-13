"""Code for hyperparameter-tuning of thresholding and LGDE in newsgroups application"""

import pickle
import numpy as np
import pandas as pd

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm

from lgde import LGDEWithCDlib, evaluate_prediction


###################################
# data loading and pre-processing #
###################################

# load fine-tuned embeddings for different dimensions
embedding_50d = pd.read_pickle(
    "data/embeddings/glove_newsgroups_15_1.0_50d_1.0mu_df.pkl"
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
newsgroups_test = fetch_20newsgroups(
    subset="test", remove=("headers", "footers", "quotes")
)

# create binary split with 1: comp, misc and sci and 0: other groups
target_labels = np.asarray([1, 2, 3, 4, 5, 6, 11, 12, 13, 14])

# define training set
X_train = pd.Series(newsgroups_train["data"])
y_train = np.asarray(np.in1d(newsgroups_train["target"], target_labels), dtype=int)

# define test set
X_test = pd.Series(newsgroups_test["data"])
y_test = np.asarray(np.in1d(newsgroups_test["target"], target_labels), dtype=int)

# compute document-term matrix for train
vectorizer = CountVectorizer(vocabulary=word_list)
word_counts_train = vectorizer.fit_transform(X_train)
dt_train = pd.DataFrame(word_counts_train.toarray(), columns=word_list)

# and for test
word_counts_test = vectorizer.fit_transform(X_test)
dt_test = pd.DataFrame(word_counts_test.toarray(), columns=word_list)

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

    ######################################################################
    # grid search for LGDE with local modularity M and R hyperparameters #
    ######################################################################

    # define number of iterations for local modularity optimisation
    n_iterations = 50
    # define range of k for CkNN
    ks = np.arange(3, 21, dtype=int)

    # initialise results for Mod R
    size_disc_modr = np.zeros((len(ks), n_iterations))
    precision_modr_train = np.zeros((len(ks), n_iterations))
    recall_modr_train = np.zeros((len(ks), n_iterations))
    fscore_modr_train = np.zeros((len(ks), n_iterations))
    precision_modr_test = np.zeros((len(ks), n_iterations))
    recall_modr_test = np.zeros((len(ks), n_iterations))
    fscore_modr_test = np.zeros((len(ks), n_iterations))
    communities_modr = []

    # initialise results for Mod M
    size_disc_modm = np.zeros((len(ks), n_iterations))
    precision_modm_train = np.zeros((len(ks), n_iterations))
    recall_modm_train = np.zeros((len(ks), n_iterations))
    fscore_modm_train = np.zeros((len(ks), n_iterations))
    precision_modm_test = np.zeros((len(ks), n_iterations))
    recall_modm_test = np.zeros((len(ks), n_iterations))
    fscore_modm_test = np.zeros((len(ks), n_iterations))
    communities_modm = []

    # create LGDEWithCDlib object
    lgde_cdlib = LGDEWithCDlib(seed_dict, word_list, word_vecs)

    # iterate through k
    for i, k in tqdm(enumerate(ks), total=len(ks)):
        # construct CkNN network
        lgde_cdlib.construct_network(k=k)

        communities_modr_k = []
        communities_modm_k = []

        # repeate optimistions
        for j in range(n_iterations):

            # expand based on local modularity R
            lgde_cdlib.detect_local_communities(method="mod_r", disable_tqdm=True)
            size_disc_modr[i, j] = len(lgde_cdlib.discovered_dict_)
            communities_modr_k.append(lgde_cdlib.semantic_communities_)
            # evaluate expanded dictionary on training set
            p, r, f = evaluate_prediction(
                lgde_cdlib.expanded_dict_, y_train, dt_train, with_report=False
            )
            precision_modr_train[i, j] = p
            recall_modr_train[i, j] = r
            fscore_modr_train[i, j] = f
            # evaluate expanded dictionary on test set
            p, r, f = evaluate_prediction(
                lgde_cdlib.expanded_dict_, y_test, dt_test, with_report=False
            )
            precision_modr_test[i, j] = p
            recall_modr_test[i, j] = r
            fscore_modr_test[i, j] = f

            # expand based on local modularity M
            lgde_cdlib.detect_local_communities(method="mod_m", disable_tqdm=True)
            size_disc_modm[i, j] = len(lgde_cdlib.discovered_dict_)
            communities_modm_k.append(lgde_cdlib.semantic_communities_)
            # evaluate expanded dictionary on training set
            p, r, f = evaluate_prediction(
                lgde_cdlib.expanded_dict_, y_train, dt_train, with_report=False
            )
            precision_modm_train[i, j] = p
            recall_modm_train[i, j] = r
            fscore_modm_train[i, j] = f
            # evaluate expanded dictionary on test set
            p, r, f = evaluate_prediction(
                lgde_cdlib.expanded_dict_, y_test, dt_test, with_report=False
            )
            precision_modm_test[i, j] = p
            recall_modm_test[i, j] = r
            fscore_modm_test[i, j] = f

        # append discovered words
        communities_modr.append(communities_modr_k)
        communities_modm.append(communities_modm_k)

    # store results
    results_modr = {
        "size": size_disc_modr,
        "precision_train": precision_modr_train,
        "recall_train": recall_modr_train,
        "fscore_train": fscore_modr_train,
        "precision_test": precision_modr_test,
        "recall_test": recall_modr_test,
        "fscore_test": fscore_modr_test,
        "communities": communities_modr,
        "ks": ks,
    }
    results_modm = {
        "size": size_disc_modm,
        "precision_train": precision_modm_train,
        "recall_train": recall_modm_train,
        "fscore_train": fscore_modm_train,
        "precision_test": precision_modm_test,
        "recall_test": recall_modm_test,
        "fscore_test": fscore_modm_test,
        "communities": communities_modm,
        "ks": ks,
    }

    results = {}
    results["modr"] = results_modr
    results["modm"] = results_modm

    with open(
        f"data/gs/gs_local_mod_{n_iterations}iter_glove_newsgroups_15_1.0_{dim}d_5.0mu_df.pkl",
        "wb",
    ) as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
