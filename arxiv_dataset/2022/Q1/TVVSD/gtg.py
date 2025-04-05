from typing import List, Tuple

import numpy as np
import pandas as pd
import torch

VERB_TYPES = {}

with open('data/labels/motion_verbs.csv') as motion_verbs:
    VERB_TYPES['motion'] = [line.rstrip('\n') for line in motion_verbs]

with open('data/labels/non_motion_verbs.csv') as non_motion_verbs:
    VERB_TYPES['nonmotion'] = [line.rstrip('\n') for line in non_motion_verbs]
 

def affinity_matrix(elements: np.ndarray) -> np.ndarray:
    """
    Compute pair-wise similarity matrix from an array; It represents the
    graph weight matrix W.

    The affinity measure is the cosine similarity.

    Args:
        elements: A 1D-vector of size n.

    Returns:
        An NxN symmetric matrix.
    """
    n_points = len(elements)
    affinity = np.zeros((n_points, n_points))  # Affinity matrix

    for i in range(n_points):
        w_i = elements[i]
        for j in range(i + 1, n_points):  # The matrix is symmetric so computations can be skipped.
            w_j = elements[j]
            w_ij = np.dot(w_i, w_j)  # Cosine similarity (since vectors are already normalised).
            affinity[i, j] = w_ij
            affinity[j, i] = w_ij
    return affinity


def strategy_space(sense_labels: pd.DataFrame, senses: pd.DataFrame, all_senses: bool = False) -> np.ndarray:
    """
    Generate the strategy space of the game, where rows are
    mixed_strategies (images representations) and columns are pure
    strategies (verb senses). Each cell is uniformly initialised, the
    cells for which verb and sense intersection is null are set to zero.

    Args:
        sense_labels: A dataframe containing the correct sense for each
            pair (image, verb).
        senses: A dataframe of verb senses.
        all_senses: if True, the sense probability of a data-point is
            distributed on all senses, otherwise they are restricted to
            the ones related to a given verb.

    Returns:
        An NxC matrix containing row-wise probability distributions.
    """
    n_points = len(sense_labels)
    n_senses = len(senses)
    strategies = np.zeros((n_points, n_senses))  # Strategy space

    for i, row in enumerate(sense_labels.itertuples()):  # Row: index of image images_captions table
        verb = getattr(row, 'lemma')

        if all_senses:
            filtered_senses = senses
        else:
            filtered_senses = senses.query('lemma == @verb')

        for j in range(len(filtered_senses)):
            col = filtered_senses.iloc[j].name  # Columns: index of sense in pandas dataframe
            strategies[i, col] = 1 / len(filtered_senses)
    return strategies


def first_sense_strategy_space(sense_labels: pd.DataFrame, senses: pd.DataFrame, alpha: float,
                               all_senses: bool = False) -> np.ndarray:
    """
    Generate the strategy space of the game, where rows are
    mixed_strategies (images representations) and columns are pure
    strategies (verb senses). The first sense probability of each verb
    is set to an alpha value, whereas the remaining C-1 cells are set to
    (1-alpha)/C. Cells for which verb and sense intersection is null are
    set to zero.

    Args:
        sense_labels: A dataframe containing the correct sense for each
            pair (image, verb).
        senses: A dataframe of verb senses.
        alpha: Value used to initialise the first sense of each verb.
        all_senses: if True, the sense probability of a data-point is
            distributed on all senses, otherwise they are restricted to
            the ones related to a given verb.

    Returns:
        An NxC matrix containing row-wise probability distributions
    """
    n_points = len(sense_labels)
    n_senses = len(senses)
    strategies = np.zeros((n_points, n_senses))  # Strategy space

    for i, row in enumerate(sense_labels.itertuples()):  # Row: index of image images_captions table
        verb = getattr(row, 'lemma')
        if all_senses:
            filtered_senses = senses
        else:
            filtered_senses = senses.query('lemma == @verb')

        for j in range(len(filtered_senses)):
            col = filtered_senses.iloc[j].name  # Columns: index of sense in pandas dataframe
            if j == 0:
                strategies[i, col] = alpha
            else:
                strategies[i, col] = (1 - alpha) / (len(filtered_senses) - 1)
    return strategies


def mfs_heuristic_strategies(sense_labels: pd.DataFrame, senses: pd.DataFrame) -> np.ndarray:
    """
    Initialise the strategy space according to the Most Frequent Sense
    of each verb sense in 'sense_labels'.

    Args:
        sense_labels: A dataframe containing the correct sense for each
            pair (image, verb).
        senses: A dataframe of verb senses.

    Returns:
        An NxC matrix composed of one-hot vectors.
    """
    n_points = len(sense_labels)
    n_senses = len(senses)
    strategies = np.zeros((n_points, n_senses))  # Strategy space

    for i, row in enumerate(sense_labels.itertuples()):
        verb = getattr(row, 'lemma')
        filtered_senses = senses.query('lemma == @verb')
        mfs_idx = sense_labels.query(
            "lemma == @verb").groupby('sense_chosen')['lemma'].count().values.argmax()

        col = filtered_senses.iloc[mfs_idx].name
        strategies[i, col] = 1
    return strategies


def player_strategy_indexing(verb: str, senses: pd.DataFrame) -> List[int]:
    """
    Returns a list of indexes to access the senses probabilities
    columns of a given verb.

    Args:
        verb: Verb used as index to get a sub-row of the strategy space
        senses: A dataframe of verb senses.

    Returns:
        A list of columns indexes.
    """
    column_indexes = senses.query("lemma == @verb").index.tolist()
    return column_indexes


def prior_knowledge(sense_labels, nodes, senses):
    """
    Initialise the probability distributions of the strategy space using
    a cosine similarity measure between the representations of images
    and the representations of senses.

    Args:
        sense_labels: A dataframe containing the correct sense for each
            pair (image, verb)
        nodes: Matrix where each row is the representation of an image
        senses: A dataframe of verb senses

        Returns:
            A new strategy space initialised with probabilities based
            on the similarity with senses
    """
    strategies = np.zeros((len(nodes), len(senses)))
    for i, row in enumerate(sense_labels.itertuples()):
        i_t = nodes[i]
        verb = getattr(row, 'lemma')
        filtered_senses = senses.query('lemma == @verb')
        dot_prod = filtered_senses['vects'].apply(lambda s_t: np.dot(i_t, s_t)).to_numpy()
        probs = dot_prod / dot_prod.sum()
        strategies[i, player_strategy_indexing(verb, senses)] = probs

    return strategies


def one_hot(senses: pd.DataFrame, verb: str, sense_num: int) -> np.ndarray:
    """
    Encode a sense as a one-hot vector.

    Args:
        senses: A dataframe of verb senses.
        verb: The verb to access a subset of senses.
        sense_num: The sense index of the verb to map.

    Returns:
        A one-hot vector with a dimensionality related to the total
        number of senses.
    """
    sense = senses.query("lemma == @verb and sense_num == @sense_num").iloc[0]
    vector = np.zeros(len(senses))
    vector[senses.index.get_loc(sense.name)] = 1
    return vector


def generate_nodes(data_points: pd.DataFrame, labels: pd.DataFrame, representation_type: str) -> np.ndarray:
    """
    Generate a matrix from a dataframe of vectors.

    Args:
        data_points: A dataframe of vectors representing encoded
            data-points.
        labels: A dataframe containing the correct sense for each
            pair (image, verb).
        representation_type: The column name to access in 'images' dataframe.

    Returns:
        A matrix in which each row represent an encoded feature.
    """
    nodes = None
    for _, row in enumerate(labels.itertuples()):
        img_name = getattr(row, 'image')
        if nodes is None:
            nodes = data_points.loc[img_name][representation_type]
        else:
            node = data_points.loc[img_name][representation_type]
            nodes = np.vstack([nodes, node])
    return nodes


def labelling(senses: pd.DataFrame, sense_labels: pd.DataFrame, seed_number: int, labels_per_sense: int) -> np.ndarray:
    """
    Randomly pick a list of elements to label for a
    Semi-supervised approach. A label for each sense is returned.

    Args:
        sense_labels: A dataframe containing the correct sense for each
            pair (image, verb).
        senses: A dataframe of verb senses.
        seed_number: A seed for reproducible experiments.
        labels_per_sense: The maximum number of labelled to use for each
            class (sense).

    Returns:
        A list of indexes of elements in 'sense_labels' to label
    """
    random_seed = np.random.RandomState(seed_number)
    labeled_indexes = []
    for _, row in enumerate(senses.itertuples()):
        verb = getattr(row, 'lemma')
        sense_num = getattr(row, 'sense_num')
        nodes = sense_labels.query("lemma == @verb and sense_chosen == @sense_num")
        if len(nodes) == 0:
            continue
        sample_size = min(labels_per_sense, len(nodes) - 1 if len(nodes) > 1 else 1)

        sample_nodes = nodes.sample(n=sample_size, random_state=random_seed)
        for i in range(sample_size):
            node_idx = sample_nodes.iloc[i].name
            labeled_indexes.append(node_idx)
    return np.array(labeled_indexes)


def replicator_dynamics(W: np.ndarray, strategies: np.ndarray, max_it: int = 100) -> np.ndarray:
    """
    Compute Nash equilibria using Replicator Dynamics.

    Args:
        W: The affinity matrix of the game.
        strategies: The strategy space of the game.
        max_it: The number of RD iterations to perform.

    Returns:
        The strategy space after the game convergence.
    """
    p = strategies.copy()
    n_iter = 0

    X = torch.from_numpy(p)
    W = torch.from_numpy(W)
    while n_iter < max_it:
        X = X * torch.matmul(W, X)
        X /= (X.sum(dim=X.dim() - 1).unsqueeze(X.dim() - 1) + 10**-15)
        n_iter += 1
    return X.numpy()


def gtg(y: pd.DataFrame, weights: np.ndarray, labeled_senses_idx: np.ndarray, strategies: np.ndarray, max_it: int,
        first_sense: bool = False) -> Tuple[float, float]:
    """
    Run GTG on a partially-labelled dataset and compute accuracy.
    Args:
        y: data-points ground truth.
        weights: pair-wise affinity matrix between data-points.
        labeled_senses_idx: indexes of labels to use.
        strategies: NxC strategy-space matrix in which each row is a
            verb and each column is a sense.
        max_it: the number of RD iterations.
        first_sense: if True RD are not run and the first sense of
            each verb is returned.

    Returns:
        The accuracy of motion verbs and non-motion verbs.

    """
    n_points = len(weights)

    probabilities = strategies.copy()

    labelled_probabilities = y['one_hot'].iloc[labeled_senses_idx]
    for i in range(len(labeled_senses_idx)):
        probabilities[labeled_senses_idx[i], :] = labelled_probabilities.iloc[i]

    if first_sense:
        p_new = probabilities
    else:
        p_new = replicator_dynamics(weights, probabilities, max_it)

    unlabeled = np.setdiff1d(np.arange(n_points), labeled_senses_idx)
    n_unlabeled = len(unlabeled)

    correct = {'motion': 0, 'nonmotion': 0}
    wrong = {'motion': 0, 'nonmotion': 0}
    y_cap = np.zeros((n_unlabeled, 1))

    # Accuracy statistics
    for i in range(n_unlabeled):
        y_cap[i] = np.argmax(p_new[unlabeled[i], :])

        if y['lemma'].iloc[unlabeled[i]] in VERB_TYPES['motion']:
            if y_cap[i] == np.argmax(y['one_hot'].iloc[unlabeled[i]]):
                correct['motion'] += 1
            else:
                wrong['motion'] += 1
        elif y['lemma'].iloc[unlabeled[i]] in VERB_TYPES['nonmotion']:
            if y_cap[i] == np.argmax(y['one_hot'].iloc[unlabeled[i]]):
                correct['nonmotion'] += 1
            else:
                wrong['nonmotion'] += 1
        else:
            raise ValueError('Unknown verb type')

    motions = (correct['motion'] / (correct['motion'] + wrong['motion']))
    non_motions = (correct['nonmotion'] / (correct['nonmotion'] + wrong['nonmotion']))

    return motions, non_motions
