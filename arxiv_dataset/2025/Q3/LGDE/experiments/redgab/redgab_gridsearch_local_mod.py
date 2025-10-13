"""Code for hyperparameter-tuning of thresholding and LGDE in redgab application"""

import pickle
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from lgde import LGDEWithCDlib, evaluate_prediction

###################################
# data loading and pre-processing #
###################################

# load fine-tuned embeddings for different dimensions
embedding_50d = pd.read_pickle("data/glove_redgab_15_0.8_50d_1.0mu_df.pkl")
embedding_100d = pd.read_pickle("data/glove_redgab_15_0.8_100d_1.0mu_df.pkl")
embedding_300d = pd.read_pickle("data/glove_redgab_15_0.8_300d_1.0mu_df.pkl")

# retrieve word vectors at different dimensions
word_vecs_50d = np.zeros(
    (embedding_50d["word_vector"].shape[0], embedding_50d["word_vector"][0].shape[0])
)
for i in range(word_vecs_50d.shape[0]):
    word_vecs_50d[i, :] = embedding_50d["word_vector"][i]

word_vecs_100d = np.zeros(
    (embedding_100d["word_vector"].shape[0], embedding_100d["word_vector"][0].shape[0])
)
for i in range(word_vecs_100d.shape[0]):
    word_vecs_100d[i, :] = embedding_100d["word_vector"][i]

word_vecs_300d = np.zeros(
    (embedding_300d["word_vector"].shape[0], embedding_300d["word_vector"][0].shape[0])
)
for i in range(word_vecs_300d.shape[0]):
    word_vecs_300d[i, :] = embedding_300d["word_vector"][i]

word_vecs_all_dimensions = [word_vecs_50d, word_vecs_100d, word_vecs_300d]

# get word list
word_list = list(embedding_50d["word_string"])

# load redgab data and split into train and test
with open("data/raw/redgab_pos_data.pkl", "rb") as handle:
    domain_data = pickle.load(handle)

X_train, X_test, y_train, y_test = train_test_split(
    domain_data.data,
    domain_data.target,
    test_size=0.25,
    stratify=domain_data.target,
    random_state=42,
)

# compute document-term matrix for train
vectorizer = CountVectorizer(vocabulary=word_list)
word_counts_train = vectorizer.fit_transform(X_train)
dt_train = pd.DataFrame(word_counts_train.toarray(), columns=word_list)

# and for test
word_counts_test = vectorizer.fit_transform(X_test)
dt_test = pd.DataFrame(word_counts_test.toarray(), columns=word_list)

# seed dictionary is defined as top 5 most frequent hate keywords according to Qian et al.
seed_dict = ["nigger", "faggot", "retard", "retarded", "cunt"]

dimension_string = ["50", "100", "300"]


###############
# grid search #
###############


for dim_ind, dim in enumerate(dimension_string):

    word_vecs = word_vecs_all_dimensions[dim_ind]

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
        f"data/gs_local_mod_{n_iterations}iter_glove_redgab_15_0.8_{dim}d_1.0mu_df.pkl",
        "wb",
    ) as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
