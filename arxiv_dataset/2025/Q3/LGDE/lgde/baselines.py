"""Code for dictionary expansion with other baseline methods."""

import functools
import itertools

from string import punctuation

import networkx as nx
import numpy as np
import pandas as pd

from cdlib import algorithms as cdlib_alg
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer

from lgde.lgde import BaseExpansion, _compute_cknn


class Thresholding(BaseExpansion):
    """Class for thresholding-based dictionary expansion.

    Parameters
    ----------
    seed_dict : list[str]
        List of words in seed dictionary.

    word_list : list[str] of length n_words
        List of all words considered.

    word_vecs : array like of shape (n_words, r)
        Word vectors with dimension r for each word in word_list.

    Attributes
    ----------
    semantic_sim : array of shape (n_words, n_words)
        Normalised cosine similarity between pairs of words.

    expanded_dict_ : list[str]
        List of words in expanded dictionary.

    discovered_dict_ : list[str]
        List of discovered keywords.

    epsilon_balls_ : dict
        Dictionary that maps seed keywords to their semantic epsilon balls.
    """

    def __init__(self, seed_dict, word_list, word_vecs):
        # initialise base class
        super().__init__(seed_dict, word_list, word_vecs)

        # attributes
        self.epsilon_balls_ = {}

    def expand(self, epsilon=0.8):
        """Method to obtain most similar words to seed dictionary via simple thresholding.

        Parameters:
        -----------
        epsilon : float
            Minimal normalised semantic similarity to discovered words, should be between
            0 and 1.
        """
        self.epsilon_balls_ = {}
        for keyword in self.seed_dict:
            keyword_ind = self.word_to_ind[keyword]
            most_similar_ind = list(
                np.where(self.semantic_sim[keyword_ind, :] > epsilon)[0]
            )
            self.epsilon_balls_[keyword] = [self.word_list[i] for i in most_similar_ind]

        # compile discovered dictionary
        self.discovered_dict_ = list(
            set(itertools.chain.from_iterable(list(self.epsilon_balls_.values())))
            - set(self.seed_dict)
        )

        # summarise expanded dictionary
        self.expanded_dict_ = self.seed_dict + self.discovered_dict_


class KNearestNeighbors(BaseExpansion):
    """Class for k-nearest neighbors (kNN) based dictionary expansion.

    Parameters
    ----------
    seed_dict : list[str]
        List of words in seed dictionary.

    word_list : list[str] of length n_words
        List of all words considered.

    word_vecs : array like of shape (n_words, r)
        Word vectors with dimension r for each word in word_list.

    Attributes
    ----------
    semantic_sim : array of shape (n_words, n_words)
        Normalised cosine similarity between pairs of words.

    expanded_dict_ : list[str]
        List of words in expanded dictionary.

    discovered_dict_ : list[str]
        List of discovered keywords.

    neighbors_ : dict
        Dictionary that maps seed keywords to their semantic epsilon balls.
    """

    def __init__(self, seed_dict, word_list, word_vecs):
        # initialise base class
        super().__init__(seed_dict, word_list, word_vecs)

        # attributes
        self.neighbors_ = {}

    def expand(self, k=1):
        """Method to obtain k most similar words for each word in seed dictionary
        via simple thresholding.

        Parameters:
        -----------
        k : int
            Number of k-most similar words.
        """
        self.neighbors_ = {}
        for keyword in self.seed_dict:
            keyword_ind = self.word_to_ind[keyword]
            most_similar_ind = np.argsort(1 - self.semantic_sim[keyword_ind, :])[
                : k + 1
            ]
            self.neighbors_[keyword] = [self.word_list[i] for i in most_similar_ind]

        # compile discovered dictionary
        self.discovered_dict_ = list(
            set(itertools.chain.from_iterable(list(self.neighbors_.values())))
            - set(self.seed_dict)
        )

        # summarise expanded dictionary
        self.expanded_dict_ = self.seed_dict + self.discovered_dict_


class IKEA(BaseExpansion):
    """Class for IKEA-based dictionary expansion [1]_.

    Parameters
    ----------
    seed_dict : list[str]
        List of words in seed dictionary.

    word_list : list[str] of length n_words
        List of all words considered.

    word_vecs : array like of shape (n_words, r)
        Word vectors with dimension r for each word in word_list.

    Attributes
    ----------
    semantic_sim : array of shape (n_words, n_words)
        Normalised cosine similarity between pairs of words.

    expanded_dict_ : list[str]
        List of words in expanded dictionary.

    discovered_dict_ : list[str]
        List of discovered keywords.

    References:
    -----------
        .. [1] Gharibshah et al., 'IKEA: Unsupervised domain-specific keyword-expansion',
        *2022 IEEE/ACM International Conference on Advances in Social Networks Analysis
        and Mining (ASONAM)*, pp. 496-503, Nov. 2022, doi: 10.1109/ASONAM55673.2022.10068656.
    """

    def expand(self, epsilon=0.8):
        """Method to obtain most similar words to seed dictionary via simple thresholding.

        Parameters:
        -----------
        epsilon : float
            Minimal normalised semantic similarity to discovered words using IKEA method,
            should be between 0 and 1.
        """

        size_expanded = 0
        ikea_dict_ind = set([self.word_to_ind[keyword] for keyword in self.seed_dict])

        # iterate as long as we can expand the expanded dictionary
        while abs(len(ikea_dict_ind) - size_expanded) > 0:
            size_expanded = len(ikea_dict_ind)
            # find all words whose mean similarity to all words in expanded dictionary is
            # larger than epsilon
            new_kws = set(
                np.argwhere(
                    self.semantic_sim[list(ikea_dict_ind)].mean(axis=0) > epsilon
                ).flatten()
            )
            # add those words to set
            ikea_dict_ind = new_kws.union(ikea_dict_ind)

        # compile expanded dictionary
        self.expanded_dict_ = [self.word_list[i] for i in ikea_dict_ind]

        # derive discovered dictionary
        self.discovered_dict_ = list(set(self.expanded_dict_) - set(self.seed_dict))


class LGDEWithCDlib(BaseExpansion):
    """Class for Local Graph-based Dictionary Expansion with alternative local community
    detection method.

    This method constructs a weighted undirected semantic similarity graph from the
    normalised cosine similarity matrix using CkNN [2]_ and computes semantic communities
    for each seed keyword using a local community detection method implemented in CDlib [3]_.

    Parameters
    ----------
    seed_dict : list[str]
        List of words in seed dictionary.

    word_list : list[str] of length n_words
        List of all words considered.

    word_vecs : array like of shape (n_words, r)
        Word vectors with dimension r for each word in word_list.

    Attributes
    ----------
    semantic_sim : array of shape (n_words, n_words)
        Normalised cosine similarity between pairs of words.

    expanded_dict_ : list[str]
        List of words in expanded dictionary.

    discovered_dict_ : list[str]
        List of discovered keywords.

    adjacency_ : array of shape (n_words, n_words)
        Adjacency matrix of undirected weighted semantic similarity graph.

    semantic_communities_ : dict
        Dictionary that maps seed keywords to their semantic communities.

    References:
    -----------
        .. [2] T. Berry and T. Sauer, 'Consistent manifold representation for
        topological data analysis', *Foundations of Data Science*, vol. 1, no. 1,
        pp. 1-38, Feb. 2019, doi: 10.3934/fods.2019001.
        .. [3] G. Rossetti, L. Milli and R. Cazabet, 'CDLIB: a python library to extract,
        compare and evaluate communities from complex networks', *Applied Network
        Science*, vol. 4, no. 1, pp. 1-26, Dec. 2019, doi: 10.1007/s41109-019-0165-9
    """

    def __init__(self, seed_dict, word_list, word_vecs):
        # initialise base class
        super().__init__(seed_dict, word_list, word_vecs)

        # attributes
        self.adjacency_ = np.array
        self.semantic_communities_ = {}
        self.nx_graph_ = None

    def construct_network(self, k=4):
        """Construct semantic similarity network using CkNN.

        Parameters:
        -----------
        k : int
            Number of nearest neighbours considered for CkNN construction, should be
            at least 1.
        """

        # compute cknn graph
        backbone = _compute_cknn(1 - self.semantic_sim, k=k, delta=1.0)
        np.fill_diagonal(backbone, 0)

        # add self-loops to singletons such that transition matrix is well defined
        for singleton in np.where(backbone.sum(axis=1) == 0)[0]:
            backbone[singleton, singleton] = 1

        backbone = np.asarray(backbone)

        # obtain similarity network
        self.adjacency_ = np.multiply(self.semantic_sim, backbone)

    def detect_local_communities(self, method="lswl", timeout=60, disable_tqdm=False):
        """Detect local semantic communities of seed keywords using LSWL/SIWO,
        which is a parameter-free method. See: https://cdlib.readthedocs.io/en/
        latest/reference/generated/cdlib.algorithms.lswl.html#cdlib.algorithms.lswl

        Parameters:
        -----------

        method : string
            Name of local community detection method implemented in CDlib including:
            - "lswl": LSWL/SIWO method [4]
            - "mod_r": Local modularity R [5]
            - "mod_m": Local modularity M [6]

        timeout : float
            The maximum time in seconds in which LSWL should retrieve the community.

        disable_tqdm : bool
            Disable progress bar for semantic community detection if True.

        References:
        -----------
        .. [4] M. Zafarmand et al., 'Fast local community discovery relying on
        the strength of links', *Social Network Analysis and Mining*, vol. 13,
        no 1., p. 112, Sep. 2023, doi:10.1007/s13278-023-01115-7.
        .. [5] A. Clauset, 'Finding local community structure in networks',
        *Physical Review E*, vol. 72, no. 2, p. 026132, Aug. 2005,
        doi: 10.1103/PhysRevE.72.026132
        .. [6] F. Luo, J. Wang and E. Promislow, 'Exploring Local Community Structures
        in Large Networks', *WI'06*, p. 233-239, Dec. 2006, doi: 10.1109/WI.2006.72
        """

        # select local community detection (LCD) method
        if method == "lswl":
            # set timeout for LSWL/SIWO
            lcd_method = functools.partial(cdlib_alg.lswl, timeout=timeout)

        elif method == "mod_r":
            lcd_method = cdlib_alg.mod_r

        elif method == "mod_m":
            lcd_method = cdlib_alg.mod_m

        else:
            raise ValueError(
                "Only lswl, mod_r and mod_m are available for local community detection in CDlib."
            )

        # construct networkx graph
        nx_graph = nx.from_numpy_array(self.adjacency_)

        for keyword in tqdm(self.seed_dict, disable=disable_tqdm):
            keyword_ind = self.word_to_ind[keyword]
            # compute local community with CDLib
            community_ind = []
            try:
                community_ind = lcd_method(nx_graph, keyword_ind).communities[0]
            except IndexError:
                # for some indices CDlib may raise an IndexError and local community
                # can't be computed
                print(f"Local community detection failed for '{keyword}'.")

            # retrieve words corresponding to community
            self.semantic_communities_[keyword] = [
                self.word_list[i] for i in community_ind
            ]

        # compile discovered dictionary
        all_communities = set(
            itertools.chain.from_iterable(list(self.semantic_communities_.values()))
        )
        self.discovered_dict_ = list(all_communities - set(self.seed_dict))

        # summarise expanded dictionary
        self.expanded_dict_ = self.seed_dict + self.discovered_dict_

    def expand(self, k=4, method="lswl", timeout=60, disable_tqdm=False):
        """Expand seed dictionary with LGDE method using a local community detection method
        is CDlib.

        Parameters:
        -----------
        k : int
            Number of nearest neighbours considered for CkNN construction, should be
            at least 1.

        method : string
            Name of local community detection method implemented in CDlib including:
            - "lswl": LSWL/SIWO method [4]
            - "mod_r": Local modularity R [5]
            - "mod_m": Local modularity M [6]

        timeout : float
            The maximum time in seconds in which LSWL should retrieve the community.

        disable_tqdm : bool
            Disable progress bar for semantic community detection if True.

        References:
        -----------
        .. [4] M. Zafarmand et al., 'Fast local community discovery relying on
        the strength of links', *Social Network Analysis and Mining*, vol. 13,
        no 1., p. 112, Sep. 2023, doi:10.1007/s13278-023-01115-7.
        .. [5] A. Clauset, 'Finding local community structure in networks',
        *Physical Review E*, vol. 72, no. 2, p. 026132, Aug. 2005,
        doi: 10.1103/PhysRevE.72.026132
        .. [6] F. Luo, J. Wang and E. Promislow, 'Exploring Local Community Structures
        in Large Networks', *WI'06*, p. 233-239, Dec. 2006, doi: 10.1109/WI.2006.72
        """
        self.construct_network(k)
        self.detect_local_communities(method, timeout, disable_tqdm)


class TextRank:
    """Class for Dictionary Expansion with TextRank [7]_.

    This method computes TextRank for a given vocabulary from documents in the corpus
    that contain at least one seed keyword.

    Parameters
    ----------
    seed_dict : list[str]
        List of words in seed dictionary.

    word_list : list[str] of length n_words
        List of all words considered.

    Attributes
    ----------
    text_rank_ : array of shape (n_words,)
        TextRank value for each word in list of words.

    expanded_dict_ : list[str]
        List of words in expanded dictionary.

    discovered_dict_ : list[str]
        List of discovered keywords.

    References:
    -----------
        .. [7] R. Mihalcea and P. Tarau, 'TextRank: Bringing Order into Text', *Proceedings
        of the 2004 Conference on Empirical Methods in Natural Language Processing*,
        pp. 404-411, Jul. 2004.
    """

    def __init__(self, seed_dict, word_list):
        # store list of words
        self.word_list = word_list

        # compute dictionary that maps words to indices
        self.word_to_ind = {self.word_list[i]: i for i in range(self.n_words)}

        # compute effective seed dictionary
        self.seed_dict = []
        for keyword in seed_dict:
            if keyword in set(self.word_list):
                self.seed_dict.append(keyword)

        # attributes
        self.expanded_dict_ = self.seed_dict
        self.discovered_dict_ = []
        self.text_rank_ = np.zeros(self.n_words)

    @property
    def n_words(self):
        """Computes total number of words."""
        return len(self.word_list)

    def compute_textrank(self, documents, window_size, disable_tqdm=False):
        """Compute TextRank for all words in vocabulary using only those documents
        that contain at least one seed keyword.

        Parameters:
        -----------

        documents : list[string]
            List of documents.

        window_size : int
            Size of context window for TextRank.

        disable_tqdm : bool
            Disable progress bar for semantic community detection if True.
        """

        # compute document-term matrix for documents
        vectorizer = CountVectorizer(vocabulary=self.word_list)
        word_counts_train = vectorizer.fit_transform(documents)
        dt_matrix = pd.DataFrame(word_counts_train.toarray(), columns=self.word_list)

        # predict document classes through simple keyword matching
        y_pred = np.sum(np.asarray(dt_matrix[self.seed_dict]), axis=1) > 0

        # get train documents that contain seed keywords
        documents_seed = list(documents[y_pred])

        # compute term co-occurrence matrix for given window size
        co_matrix_seed = np.zeros((self.n_words, self.n_words))
        # iterate through documents
        for doc in tqdm(documents_seed, disable=disable_tqdm):

            # replace line breaks
            doc = doc.replace("\n", " ")

            # remove white space
            doc = doc.strip()

            # remove punctuation
            doc = doc.translate(str.maketrans("", "", punctuation))

            # split document into word sequence
            word_sequence = doc.split()

            # iterate through words
            for i, word_i in enumerate(word_sequence):
                # check if word is part of vocabulary
                if word_i in self.word_list:
                    # look at all words within context window:
                    for j in range(i - window_size - 1, i + window_size + 1):
                        if j >= 0 and not j == i and j < len(word_sequence):
                            word_j = word_sequence[j]
                            # check if word within context window is part of vocabulary
                            if word_j in self.word_list:
                                # increase co-occurence count by one
                                co_matrix_seed[
                                    self.word_to_ind[word_i], self.word_to_ind[word_j]
                                ] += 1

        # create networkx graph from co-occurrence adjacency matrix
        graph_co_seed = nx.from_numpy_array(co_matrix_seed)

        # free up memory
        del co_matrix_seed

        # TextRank is computed as PageRank in co-occurrence graph
        tr = nx.pagerank(graph_co_seed, alpha=0.85)

        # free up memory
        del graph_co_seed

        # convert to numpy array
        self.text_rank_ = np.array(list(tr.values()))

    def expand(self, n_top=1):
        """Expand seed dictionary by words with highest TextRank.

        Parameters:
        -----------

        n_top : int
            Number of top-ranked words to be discovered.
        """

        # get top ranked indices
        top_rank_ind = np.argsort(-self.text_rank_)[:n_top]
        # get top ranked words
        top_words = [self.word_list[ind] for ind in top_rank_ind]
        # compile dictionaries
        self.discovered_dict_ = list(set(top_words) - set(self.seed_dict))
        self.expanded_dict_ = self.seed_dict + self.discovered_dict_
