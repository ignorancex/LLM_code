"""Code for dictionary expansion with LGDE."""

import itertools
import random

import igraph as ig
import networkx as nx
import numpy as np

from scipy.spatial.distance import pdist, squareform
from severability import transition_matrix, node_component
from tqdm import tqdm

from lgde.plotting import _plot_semantic_network, _plot_semantic_communities


class BaseExpansion:
    """Base class for dictionary expansion.

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
    """

    def __init__(self, seed_dict, word_list, word_vecs):

        # store list of words
        self.word_list = word_list

        # compute dictionary that maps words to indices
        self.word_to_ind = {self.word_list[i]: i for i in range(self.n_words)}

        # compute effective seed dictionary
        self.seed_dict = []
        for keyword in seed_dict:
            if keyword in set(self.word_list):
                self.seed_dict.append(keyword)

        # compute normalised cosine similarity
        self.semantic_sim = self._compute_semantic_sim(word_vecs)

        # attributes
        self.expanded_dict_ = self.seed_dict
        self.discovered_dict_ = []

    @property
    def n_words(self):
        """Computes total number of words."""
        return len(self.word_list)

    @property
    def n_seed(self):
        """Computes number of seed keywords."""
        return len(self.seed_dict)

    @property
    def n_discovered(self):
        """ "Computes number of discovered words."""
        return len(self.discovered_dict_)

    def _compute_semantic_sim(self, word_vecs):
        """Computes normalised cosine similarity."""

        # compute normalised cosine distance and similarity
        distance = squareform(pdist(np.array(word_vecs), metric="cosine"), checks=True)
        distance = np.nan_to_num(distance)
        return 1 - distance / np.amax(distance)

    def pairwise_word_similarity(self, kw1, kw2):
        """Compute pairwise similarity between two words."""
        return self.semantic_sim[self.word_to_ind[kw1], self.word_to_ind[kw2]]


class LGDE(BaseExpansion):
    """Class for Local Graph-based Dictionary Expansion.

    This method constructs a weighted undirected semantic similarity graph from the
    normalised cosine similarity matrix using CkNN [1]_ and computes semantic communities
    for each seed keyword using local community detection with Severability [2]_.

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

    nx_graph_ : networkx graph
        NetworkX graph mainly used for drawing the semantic similarity graph.

    References:
    -----------
        .. [1] T. Berry and T. Sauer, 'Consistent manifold representation for
        topological data analysis', *Foundations of Data Science*, vol. 1, no. 1,
        pp. 1-38, Feb. 2019, doi: 10.3934/fods.2019001.
        .. [2] Y. Yu William, D. Jean-Charles, S. Yaliraki and M. Barahona, 'Severability
        of mesoscale components and local time scales in dynamical networks',
        arXiv: 2006.02972, Jun. 2020, doi: 10.48550/arXiv.2006.02972
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

    def detect_local_communities(self, t=1, disable_tqdm=False):
        """Detect local semantic communities of seed keywords using Severability.

        Parameters:
        -----------
        t : int
            Markov scale t of Severability, should be at least 1.

        disable_tqdm : bool
            Disable progress bar for semantic community detection if True.
        """

        # compute transition matrix, must be a np.matrix to work properly!
        P = transition_matrix(np.matrix(self.adjacency_))

        for keyword in tqdm(self.seed_dict, disable=disable_tqdm):
            self.semantic_communities_[keyword] = self._compute_local_community(
                keyword, P, t
            )

        # compile discovered dictionary
        all_communities = set(
            itertools.chain.from_iterable(list(self.semantic_communities_.values()))
        )
        self.discovered_dict_ = list(all_communities - set(self.seed_dict))

        # summarise expanded dictionary
        self.expanded_dict_ = self.seed_dict + self.discovered_dict_

    def expand(self, k=4, t=1, disable_tqdm=False):
        """Expand seed dictionary with LGDE method.

        Parameters:
        -----------
        k : int
            Number of nearest neighbours considered for CkNN construction, should be
            at least 1.

        t : int
            Markov scale t of Severability, should be at least 1.

        disable_tqdm : bool
            Disable progress bar for semantic community detection if True.
        """
        self.construct_network(k)
        self.detect_local_communities(t, disable_tqdm)

    def construct_nx_graph(self):
        """Construct networkx graph for semantic similarity graph."""

        # remove self loops for plot
        adjacency = self.adjacency_.copy()
        np.fill_diagonal(adjacency, 0)

        # define networkx graph (undirected)
        self.nx_graph_ = nx.from_numpy_array(np.around(adjacency, 2))

        # define igraph and compute layout
        g = ig.Graph.from_networkx(self.nx_graph_)
        random.seed(1234)
        layout_drl = g.layout(layout="drl")

        # transform igraph layout to networkx layout
        graph_layout = dict()
        for i, coordinates in enumerate(list(layout_drl)):
            graph_layout[i] = np.asarray(coordinates)

        nx.set_node_attributes(self.nx_graph_, graph_layout, "pos")

    def plot_semantic_network(
        self,
        doc_freq,
        n_top=15,
        lcc_only=False,
        node_size_factor=0.1,
        alpha=0.7,
        edge_width=0.15,
        plot_with_other_words=False,
    ):
        """Plot semantic network.

        Parameters:
        -----------
        doc_freq : list
            List of document frequencies for each word.

        n_top : int
            Number of top frequent words to be annotated in graph.

        lcc_only : bool
            Only plot the largest connected component if True.

        node_size_factor : float
            Factor to determine size of nodes.

        alpha : float
            Transparency of nodes.

        edge_width : float
            Edge width.

        plot_with_other_words : bool
            Plot most frequent words which are not seed or discovered words if True.
        """
        if self.nx_graph_ is None:
            self.construct_nx_graph()

        ax = _plot_semantic_network(
            self.nx_graph_,
            self.word_to_ind,
            self.seed_dict,
            self.discovered_dict_,
            doc_freq,
            n_top,
            lcc_only,
            node_size_factor,
            alpha,
            edge_width,
            plot_with_other_words,
        )

        return ax

    def plot_semantic_communities(
        self, node_size=40, n_plots=-1, figsize=(5, 5), path=None
    ):
        """Plot local semantic communities.

        Parameters:
        -----------
        node_size: float
            Size of nodes.

        n_plots : int
            Number of semantic communities to plot, if -1 plot all.

        path : str
            Path to store plots. Plots are not stored if None.
        """
        if n_plots < 0:
            n_plots = self.n_seed

        if self.nx_graph_ is None:
            self.construct_nx_graph()

        _plot_semantic_communities(
            self.nx_graph_,
            self.seed_dict,
            self.semantic_communities_,
            self.word_to_ind,
            node_size,
            n_plots=n_plots,
            figsize=figsize,
            path=path,
        )

    def _compute_local_community(self, keyword, P, t):
        keyword_ind = self.word_to_ind[keyword]

        # check if keyword is singleton in graph
        if (
            np.sum(self.adjacency_[keyword_ind])
            == self.adjacency_[keyword_ind, keyword_ind]
        ):
            return [keyword]
        # otherwise compute local community with Severability
        else:
            community_ind = node_component(P=P, i=keyword_ind, t=t)[0]
            # if keyword was found as Severability orphan, return keyword
            if len(community_ind) == 0:
                return [keyword]
            # otherwise return local community
            else:
                return [self.word_list[i] for i in community_ind]


def _compute_cknn(D, k=5, delta=1.0):
    """Compute CkNN graph."""
    # obtain rescaled distance matrix, see CkNN paper
    darray_n_nbrs = np.partition(D, k)[:, [k]]
    ratio_matrix = D / np.sqrt(darray_n_nbrs.dot(darray_n_nbrs.T))
    # threshold rescaled distance matrix by delta and return unweighted graph
    return np.array(ratio_matrix < delta, dtype=int)
