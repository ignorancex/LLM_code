import pickle
import numpy as np
import pandas as pd
import scipy.sparse as sp
import networkx as nx

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm

from lgde import TextRank, evaluate_prediction


embedding_50d = pd.read_pickle(
    "data/embeddings/glove_newsgroups_15_1.0_50d_5.0mu_df.pkl"
)
word_vecs_50d = np.zeros(
    (embedding_50d["word_vector"].shape[0], embedding_50d["word_vector"][0].shape[0])
)
for i in range(word_vecs_50d.shape[0]):
    word_vecs_50d[i, :] = embedding_50d["word_vector"][i]

# get word list
word_list = list(embedding_50d["word_string"])

# get document frequency
doc_freq = list(embedding_50d["doc_frequency"])

# fetch train and test data
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

# seed dictionary is defined as names of positive groups
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

############
# TextRank #
############

# define range of context windows
contex_windows = np.arange(2, 11)
# define range of top ranks
n_top_ranks = np.arange(0, 201)

# initialise results
size = np.zeros((len(contex_windows), len(n_top_ranks)))
precision = size.copy()
recall = size.copy()
fscore = size.copy()
discovered_dicts = []

# iterate through context windows
for i, cw in enumerate(contex_windows):

    # compute textranks for cw
    textrank = TextRank(seed_dict, word_list)
    textrank.compute_textrank(X_train, window_size=cw)

    discovered_dicts_cw = []

    for j, n in tqdm(enumerate(n_top_ranks), total=len(n_top_ranks)):
        # expand by top n words
        textrank.expand(n_top=n)
        # compute size of discovered dictionary
        size[i, j] = len(textrank.discovered_dict_)
        # store discovered dictionary
        discovered_dicts_cw.append(textrank.discovered_dict_)
        # evaluate expanded dictionary on training set
        p, r, f = evaluate_prediction(
            textrank.expanded_dict_, y_train, dt_train, with_report=False
        )
        precision[i, j] = p
        recall[i, j] = r
        fscore[i, j] = f

    discovered_dicts.append(discovered_dicts_cw)

results_tr = {
    "context_windows": contex_windows,
    "n_top_ranks": n_top_ranks,
    "size": size,
    "precision": precision,
    "recall": recall,
    "fscore": fscore,
    "discovered_dicts": discovered_dicts,
}

with open("data/gs/gs_textrank_newsgroups.pkl", "wb") as handle:
    pickle.dump(results_tr, handle, protocol=pickle.HIGHEST_PROTOCOL)
