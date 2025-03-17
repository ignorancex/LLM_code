import numpy as np
import scipy.stats
import ete3


class TreeLabeling:
    def __init__(self, tree_ete3: ete3.Tree, original_to_new_labeling: dict[str, str] | str):
        self.tree = tree_ete3
        self.cophenetic_matrix_of_leaves, self.leaves = self.tree.cophenetic_matrix()
        self.language_to_leaf_index = {lang: ind for ind, lang in enumerate(self.leaves)}
        self.original_to_new = {}
        self.new_to_original = {}
        self.set_labeling(original_to_new_labeling)

    def cophenetic_distance(self, a, b):
        index_a = self.language_to_leaf_index[self.new_to_original[a]]
        index_b = self.language_to_leaf_index[self.new_to_original[b]]
        return self.cophenetic_matrix_of_leaves[index_a][index_b]

    def set_labeling(self, original_to_new_labeling: dict[str, str] | str):
        if original_to_new_labeling == 'random':
            self.original_to_new = self.generate_random_labeling()
        else:
            if not self._check_labeling(original_to_new_labeling):
                raise ValueError('The provided labeling does not match the leaves of the tree.')
            self.original_to_new = {k: v for k, v in original_to_new_labeling.items()}
        self.new_to_original = {v: k for k, v in self.original_to_new.items()}

    def flip_labels(self, a, b):
        pre_a = self.new_to_original[a]
        pre_b = self.new_to_original[b]
        self.new_to_original[a] = pre_b
        self.new_to_original[b] = pre_a
        self.original_to_new[pre_a] = b
        self.original_to_new[pre_b] = a

    def build_relabeled_tree(self):
        tree = self.tree.copy()
        for leaf in tree.iter_leaves():
            leaf.name = self.original_to_new[leaf.name]
        return tree

    def build_permutation_matrix(self) -> np.array:
        """Return matrix P such that if the original tree is A, the current labeling
        is P^T A P. That is, P represents new-to-original permutation."""
        mat = np.zeros((len(self.leaves), len(self.leaves)), dtype=int)
        for new, original in self.new_to_original.items():
            mat[self.language_to_leaf_index[original], self.language_to_leaf_index[new]] = 1
        return mat

    def generate_random_labeling(self):
        return {k: v for k, v in zip(self.leaves, np.random.permutation(self.leaves))}

    def _check_labeling(self, original_to_new_labeling):
        """Return true if the both keys and values of the dictionary as sets are exactly the leaves of the tree"""
        keys = sorted(original_to_new_labeling.keys())
        values = sorted(original_to_new_labeling.values())
        leaves = sorted(self.leaves)
        return keys == leaves and values == leaves


class DistanceMatrixLabeled:
    def __init__(self, matrix: np.array, labels: list):
        if matrix.shape != (len(labels), len(labels)):
            raise ValueError('The labels of the distance matrix do not match the shape of the distance matrix.')
        if len(labels) != len(set(labels)):
            raise ValueError('The labels are not unique.')
        self.matrix = matrix
        self.labels = labels
        self.label_to_index = {lab: ind for ind, lab in enumerate(labels)}

    def distance(self, a, b):
        return self.matrix[self.label_to_index[a], self.label_to_index[b]]


def correlation_tree_labeling_distance_matrix(tree_labeling: TreeLabeling, distance_matrix: DistanceMatrixLabeled):
    """Return the Spearman correlation coefficient of the cophenetic distance of the given tree
    and the entries of the given distance matrix."""
    if sorted(tree_labeling.leaves) != sorted(distance_matrix.labels):
        raise ValueError('The labels of the tree and the distance matrix do not agree.')

    tree_values, distance_matrix_values = zip(*[(tree_labeling.cophenetic_distance(a, b),
                                                 distance_matrix.distance(a, b))
                                                for i, a in enumerate(tree_labeling.leaves)
                                                for b in tree_labeling.leaves[:i]])

    return scipy.stats.spearmanr(tree_values, distance_matrix_values).statistic
