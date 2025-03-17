import io
import random
import Bio.Phylo
import Bio.Phylo.TreeConstruction
import ete3
import numpy as np


# def load_pd_distance_matrix_to_BioDistanceMatrix(filename):
#     with open(filename, 'r') as file:
#         langs = file.readline().strip().split()
#         mat = [[0.0]] + [[float(num) for num in row.strip().split()] + [0.0] for row in file]
#     return Bio.Phylo.TreeConstruction.DistanceMatrix(langs, mat)


def convert_labeled_matrix_to_bio_phylo_distance_matrix(matrix, labels):
    matrix_as_lists = [list(row[:i+1]) for i, row in enumerate(matrix)]
    return Bio.Phylo.TreeConstruction.DistanceMatrix(labels, matrix_as_lists)


def convert_tree_bio_phylo_to_ete(bio_phylo_tree):
    buffer = io.StringIO()
    Bio.Phylo.write(bio_phylo_tree, buffer, format='newick')
    tree_newick = buffer.getvalue().strip()
    buffer.close()
    return ete3.Tree(tree_newick, format=1)


def build_tree_from_labeled_matrix(matrix, labels, algorithm):
    """Construct an ete3 tree from a given matrix and labels. Only the lower-triangular
    part of the matrix is used. The available algorithms are 'nj' and 'upgma'."""
    bio_phylo_dm = convert_labeled_matrix_to_bio_phylo_distance_matrix(matrix, labels)
    tree_constructor = Bio.Phylo.TreeConstruction.DistanceTreeConstructor()
    if algorithm == 'nj':
        return convert_tree_bio_phylo_to_ete(tree_constructor.nj(bio_phylo_dm))
    if algorithm == 'upgma':
        return convert_tree_bio_phylo_to_ete(tree_constructor.upgma(bio_phylo_dm))
    raise ValueError(f"Only 'nj' and 'upgma' algorithms are available.")


def ete3_tree_drop_distances(ete3_tree):
    ete3_tree.dist = 1
    for child in ete3_tree.children:
        ete3_tree_drop_distances(child)


def ete3_permute_leafs(ete3_tree):
    """Permute the leafs of the given tree (in place)."""
    leaves = ete3_tree.get_leaf_names()
    perm = {k: v for k, v in zip(leaves, np.random.permutation(leaves))}
    for leaf in ete3_tree.iter_leaves():
        leaf.name = perm[leaf.name]


def ete3_tree_height(tree):
    """Compute the height of the tree (the largest length of a path to a leaf)"""
    if tree.is_leaf():
        return 0
    return max(ete3_tree_height(child) for child in tree.get_children()) + 1


def generate_random_ete3_tree(leaf_labels):
    tree_dict = {}
    top_nodes = []
    for label in leaf_labels:
        tree = ete3.Tree(name=label)
        tree_dict[label] = tree
        top_nodes.append(label)
    counter = 1
    while len(top_nodes) >= 2:
        node1 = random.choice(top_nodes)
        top_nodes.remove(node1)
        node2 = random.choice(top_nodes)
        top_nodes.remove(node2)
        name = f'internal_{counter}'
        counter+= 1
        tree = ete3.Tree(name='')
        tree_dict[name] = tree
        top_nodes.append(name)
        # noinspection PyTypeChecker
        tree.add_child(tree_dict[node1])
        # noinspection PyTypeChecker
        tree.add_child(tree_dict[node2])
    return tree_dict[top_nodes[0]]


def generate_random_distance_matrix_eucl(size, dimension_of_sample=50):
    """Sample points in Euclidean space, and return their distance matrix"""
    points = np.random.random(size=(size, dimension_of_sample))
    matrix = np.zeros((size, size), dtype=float)
    for i in range(size):
        for j in range(i):
            distance = np.linalg.norm(points[i] - points[j])
            matrix[i, j] = distance
            matrix[j, i] = distance

    return matrix


class TreeDistanceCalculator:
    def __init__(self, treedist_r_package, ape_r_package):
        """As parameters, pass the rpy2 imported R packages TreeDist and ape."""
        self.treedist = treedist_r_package
        self.ape = ape_r_package

    def convert_tree_ete_to_Rphylo(self, ete_tree, trash_edge_distances=True):
        """Convert a given ete3 tree to R phylo tree (type rpy2.robjects.vectors.ListVector)."""
        newick_format = 9 if trash_edge_distances else 0
        return self.ape.read_tree(text= ete_tree.write(format=newick_format))

    def compute_distances(self, tree_1, tree_2):
        """Compute various tree distances and return as a dictionary."""
        distance_data = {}
        if isinstance(tree_1, ete3.Tree):
            tree_1 = self.convert_tree_ete_to_Rphylo(tree_1)
        if isinstance(tree_2, ete3.Tree):
            tree_2 = self.convert_tree_ete_to_Rphylo(tree_2)
        distance_data['dist_rf'] = self.treedist.RobinsonFoulds(tree_1, tree_2, normalize=True)[0]
        distance_data['dist_jrf_k1'] = self.treedist.JaccardRobinsonFoulds(tree_1, tree_2, normalize=True, k=1)[0]
        distance_data['dist_jrf_k2'] = self.treedist.JaccardRobinsonFoulds(tree_1, tree_2, normalize=True, k=2)[0]
        distance_data['dist_phylo'] = self.treedist.DifferentPhylogeneticInfo(tree_1, tree_2, normalize=True)[0]
        distance_data['dist_clust'] = self.treedist.ClusteringInfoDistance(tree_1, tree_2, normalize=True)[0]
        distance_data['dist_match'] = self.treedist.MatchingSplitDistance(tree_1, tree_2)[0]
        # distance_data['dist_kc'] = self.treedist.KendallColijn(tree_1, tree_2)[0]  # takes a long time
        distance_data['dist_path'] = self.treedist.PathDist(tree_1, tree_2)[0]
        return distance_data


def main():
    tr1 = generate_random_ete3_tree(['a', 'b', 'c', 'd', 'e', 'f', 'g'])
    print(tr1)
    print()
    tr2 = generate_random_ete3_tree(['a', 'b', 'c', 'd', 'e', 'f', 'g'])
    print(tr2)
    print(tr1.compare(tr2))


if __name__ == '__main__':
    main()