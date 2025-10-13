import pickle
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from lgde import TextRank, evaluate_prediction


embedding_50d = pd.read_pickle("data/glove_redgab_15_0.8_50d_1.0mu_df.pkl")
word_vecs_50d = np.zeros(
    (embedding_50d["word_vector"].shape[0], embedding_50d["word_vector"][0].shape[0])
)
for i in range(word_vecs_50d.shape[0]):
    word_vecs_50d[i, :] = embedding_50d["word_vector"][i]

# get word list
word_list = list(embedding_50d["word_string"])

# get document frequency
doc_freq = list(embedding_50d["doc_frequency"])

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

# seed dictionary is defined as top 5 most frequent hate keywords according to Qian et al.
seed_dict = ["nigger", "faggot", "retard", "retarded", "cunt"]

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
    textrank.compute_textrank(pd.Series(X_train, dtype=str), window_size=cw)

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

with open("data/gs_textrank_redgab.pkl", "wb") as handle:
    pickle.dump(results_tr, handle, protocol=pickle.HIGHEST_PROTOCOL)
