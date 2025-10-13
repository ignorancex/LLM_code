"""Code for hyperparameter-tuning of LGDE in 4chan application"""

import pickle
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm

from lgde import LGDE, evaluate_prediction

###################################
# data loading and pre-processing #
###################################

# load train data
corpus_train = pd.read_csv("data/4chan_train.csv", delimiter=",", encoding="utf-8")
X_train = corpus_train["text"]
y_train = np.asarray(corpus_train["v1"])

# load seed keywords
df_keywords = pd.read_csv("data/4chan_seed.csv", encoding="utf-8")
seed_dictionary = list(df_keywords["keyword_ger"].dropna())
# combine phrases by underscores
seed_dictionary_underscored = [kw.replace(" ", "_") for kw in seed_dictionary]

# load vocabulary
df_vocabulary = pd.read_pickle("data/4chan_embeddings_100d.pkl")
df_vocabulary.reset_index(inplace=True, drop=True)

# combine phrases with underscore
word_string_original = df_vocabulary["word_string"]
word_string_underscored = [
    df_vocabulary.word_string[i].replace(" ", "_") for i in range(len(df_vocabulary))
]
df_vocabulary["word_string"] = word_string_underscored

# store word vectors in matrix
W = np.zeros(
    (df_vocabulary["word_vector"].shape[0], df_vocabulary["word_vector"][0].shape[0])
)
for i in range(W.shape[0]):
    W[i, :] = df_vocabulary["word_vector"][i]

# create dictionary for phrases
word_string_underscored_dict = {}
for i in range(len(df_vocabulary)):
    word = word_string_original[i]
    if " " in word:
        word_string_underscored_dict[word] = word.replace(" ", "_")

# make train data lower case
X_train_underscored = X_train.str.lower()

# combine phrases in train data with underscore
for old, new in word_string_underscored_dict.items():

    X_train_underscored = X_train_underscored.str.replace(old, new, regex=False)

# compute document-term matrix for train data
vectorizer = CountVectorizer(vocabulary=df_vocabulary["word_string"])
word_counts_train = vectorizer.fit_transform(np.asarray(X_train_underscored)).toarray()
dt_train = pd.DataFrame(word_counts_train, columns=df_vocabulary["word_string"])

# summarise data required for LGDE
seed_dict = seed_dictionary_underscored
word_list = word_string_underscored
word_vecs = W

########################################
# grid search for LGDE hyperparameters #
########################################

# define parameters for grid search
times = np.arange(1, 11, dtype=int)
ks = np.arange(3, 21, dtype=int)

# initialise results
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
        # evaluate discovered dictionary on training set
        p, r, f = evaluate_prediction(
            lgde.discovered_dict_, y_train, dt_train, with_report=False
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

with open("data/4chan_lgde_gridsearch.pkl", "wb") as handle:
    pickle.dump(results_lgde, handle, protocol=pickle.HIGHEST_PROTOCOL)
