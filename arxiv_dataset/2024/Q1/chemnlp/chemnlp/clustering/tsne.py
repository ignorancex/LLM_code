"""Module to perform tsne given csv file."""
import matplotlib.pyplot as plt

import pandas as pd

from tqdm import tqdm
from collections import Counter
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
from time import time
import nltk

nltk.download("punkt")
# samples=len(df)
# import numpy as np
# import os
# from jarvis.db.figshare import data
# import requests
# import matplotlib.colors
# import matplotlib.cm as cm
# import feedparser
# import time
# import json


def abstract2df(df=None, idx=None, text=""):
    ps = PorterStemmer()
    """
    USAGE: Use Porter Stemming to process a paper abstract.

    ARGUMENT:
    df - DataFrame with keys ["term", "title", "summary", "id"]
    idx - index locating paper within DataFrame

    RETURNS: stem_df - DataFrame with keys ["stems", "counts", "docs"]
    """
    abstract = df.iloc[idx][text]
    # abstract = df.iloc[idx].summary
    words = word_tokenize(abstract)
    stems = [ps.stem(w).lower() for w in words]
    counts = Counter(stems)
    docs = list(np.ones(len(counts)).astype(int))
    stem_df = pd.DataFrame(
        {
            "stems": list(counts.keys()),
            "counts": list(counts.values()),
            "docs": docs,
        }
    )
    stem_df = stem_df[["stems", "counts", "docs"]]
    return stem_df


def docs2idf(df, num):
    """
    USAGE:
    Get an inverse document frequency (idf) measure for each stem,
    and add that measure to the word stem DataFrame.

    ARGUMENTS:
    df - word stem DataFrame
    num - total number of papers in document pool
    """
    df["idf"] = np.log(num / df["docs"])
    # print ('df',df)
    return df


def reduce_dim(matrix, dims):
    """
    USAGE: Perform Truncated SVD to reduce dimensionality of embedding space.

    ARGUMENTS:
    matrix - training data (sparse matrix) with shape (n_features, n_samples)
    dims - number of dimensions to project data into

    RETURNS:
    X_new - array of reduced dimensionality
    """

    tic = time()
    print("Performing Truncated SVD...")
    svd = TruncatedSVD(n_components=dims, random_state=1)
    X = svd.fit_transform(matrix)  # matrix.shape = (n_samples, n_features)
    toc = time()
    print(
        "Embeddings reduced from %d to %d using TruncatedSVD. (Time: %.2f s)"
        % (matrix.shape[1], dims, (toc - tic))
    )
    return X


def TSNE2D(X):
    """
    USAGE: Perform TSNE to reduce embedding space to 2D
    ARGUMENT: X - high-dimensional training array (n_samples, n_features ~ 100)
    RETURNS: X_embedded - 2D matrix (n_samples, 2)
    """

    tic = time()
    print("Performing TSNE...")
    X_embedded = TSNE(
        n_components=2, perplexity=30, random_state=1
    ).fit_transform(X)
    toc = time()
    print(
        "Embeddings reduced to 2 dimensions through TSNE. (Time: %.2f s)"
        % (toc - tic)
    )
    return X_embedded


def tsne(df=None, category_key="term", text="title", filename=None, ndim=128):
    print("Counting word stems in %d abstracts..." % len(df))
    samples = len(df)
    full_stem_df = pd.DataFrame()
    for idx in range(len(df)):
        # print(idx)
        stem_df = abstract2df(df=df, idx=idx, text=text)
        full_stem_df = pd.concat([full_stem_df, stem_df])
        if idx and idx % 10 == 0 or idx == len(df) - 1:
            full_stem_df = full_stem_df.groupby("stems").sum()
            full_stem_df["stems"] = full_stem_df.index
            full_stem_df = full_stem_df[["stems", "counts", "docs"]]
    full_stem_df.sort_values("counts", ascending=False, inplace=True)
    all_stems = full_stem_df  # count_stems(dfb)
    all_stems = docs2idf(all_stems, len(df))
    all_stems.reset_index(drop=True, inplace=True)
    # df=dfb
    stems = all_stems
    # assert samples <= len(df)
    idf = np.array(list(stems["idf"]))
    stems = stems[["stems", "idf"]]
    matrix = np.zeros((samples, len(stems)))

    info = []
    print("Embedding paper abstracts...")
    i = 0
    for idx in tqdm(range(samples)):
        # print(idx)
        information = list(df.iloc[idx][["id", category_key, text]])
        info.append(information)
        new_df = pd.merge(
            stems,
            abstract2df(df=df, idx=idx, text=text),
            on="stems",
            how="left",
        ).fillna(0)
        vec = np.array(list(new_df["counts"]))
        vec /= np.sum(vec)  # components sum to 1
        vec *= idf  # apply inverse document frequency
        matrix[i, :] = vec
        i += 1
    print("Paper abstracts converted to vectors.")

    matrix = np.nan_to_num(matrix, nan=0.0)
    X = reduce_dim(matrix, ndim)
    X_embedded = TSNE2D(X)

    # %matplotlib inline
    plt.rcParams.update({"font.size": 14})
    plt.figure(figsize=(8, 8))
    X = X_embedded
    x = X[:, 0]
    y = X[:, 1]

    term_list = list(np.array(info).T[1])
    term_set = list(set(term_list))
    term_list = [term_set.index(term) for term in term_list]

    color_list = plt.cm.tab10(term_list)

    lbls = []
    xyz = []
    for i, j, k, p in zip(x, y, term_list, color_list):
        if k not in lbls:
            lbls.append(k)
            xyz.append([i, j, k])
            plt.scatter(i, j, s=10, c=p, label=term_set[k])

    plt.scatter(x, y, s=10, c=color_list)  # ,label=term_list)
    plt.legend(loc="lower left")
    # plt.xlim([-40,40])
    # plt.ylim([-40,40])
    plt.xticks([])
    plt.yticks([])
    if filename is not None:
        plt.savefig(filename)
        plt.close()
    plt.show()


# plot_2D(X_embedded, info)
# plt.savefig('tsne.png')
# plt.close()

if __name__ == "__main__":
    df = pd.read_csv("id_term_title.csv")
    tsne(df=df, filename="x.png")
