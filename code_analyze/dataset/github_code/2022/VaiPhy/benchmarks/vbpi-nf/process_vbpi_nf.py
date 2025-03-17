######################################################################################
#
# Authors :
#
#   Email :
#
# process_vbpi_nf: Functions to process Vbpi-Nf outputs.
#####################################################################################

import sys
sys.path.append('../../')
import numpy as np
from ete3 import Tree
from phylo_tree import PhyloTree
from utils import compute_loglikelihood
from data_utils import read_nex_file
from tree_utils import newick2nx, update_topology, save_tree_image


def load_tree(newick_str, n_taxa, branch_lengths=None):
    """ This function processes Vbpi-Nf tree, assigns lengths to the corresponding edges
        & returns tree in NetworkX format.
    :param newick_str: Newick string
    :param n_taxa: Number of taxa
    :param branch_lengths: Branch lengths in normal scale, in postorder list
    :return: NetworkX tree.
    """

    tree_ete = Tree(newick_str, format=3)

    for node in tree_ete.traverse("postorder"):
        # Add the root's name
        if node.name == "":
            node.add_feature("name", 2 * n_taxa - 3)

        # Update branch lengths
        if branch_lengths is not None:
            for child in node.children:
                child.add_feature("dist", branch_lengths[int(child.name)])

    new_newick_str = tree_ete.write(format=3)
    tree_nx = newick2nx(new_newick_str, n_taxa)
    tree_nx = update_topology(tree_nx, 2 * n_taxa - 3)

    return tree_nx, new_newick_str


def load_particle_trees(n_particles, n_taxa, tree_fname_prefix, log_branch_lengh_fname=None):
    """ This function processes multiple Vbpi-Nf trees, assigns lengths to the corresponding edges
            & returns trees as a list of NetworkX trees.
        :param n_particles: Number of particles
        :param n_taxa: Number of taxa
        :param tree_fname_prefix: The prefix of tree filenames. (i.e "../particle_tree_")
        :param log_branch_lengh_fname: Branch lengths filename. (Numpy array of size n_particles x n_edges)
        :return: list of NetworkX trees.
        """

    # Load branch lengths, if available
    if log_branch_lengh_fname is not None:
        norm_branch_lengths = np.exp(np.load(log_branch_lengh_fname))
    else:
        norm_branch_lengths = None

    # Load particle trees one by one
    processed_trees = []
    processed_trees_newicks = []

    for p in range(n_particles):
        fname = tree_fname_prefix + str(p) + ".nw"
        with open(fname) as f:
            tree_newick = f.read()
        tree_nx, tree_newick = load_tree(tree_newick, n_taxa, branch_lengths=norm_branch_lengths[p])

        processed_trees.append(tree_nx)
        processed_trees_newicks.append(tree_newick)

    return processed_trees, processed_trees_newicks


def compare_log_likelihoods(particle_trees, data, vbpi_nf_loglik, particle_trees_newicks):
    phylo = PhyloTree(data)

    for p in range(len(particle_trees)):
        up_table = phylo.compute_up_messages(particle_trees[p])
        vaiphy_loglik = compute_loglikelihood(up_table)

        print("Tree: ", p, "\tVbpi Log-lik: ", vbpi_nf_loglik[p], "\tVaiPhy Log-lik: ", vaiphy_loglik,
              "\tNewick: ", particle_trees_newicks[p])


'''
print("Hello, world!")

n_particles = 10
n_taxa = 27
iter_no = 400000

data_dir = "data/"
res_dir = "results/"

exp_name = "data_seed_1_taxa_20_pos_300"
prefix = "base_2021-05-11.pt_iter_" + str(iter_no) + "_"

#exp_name = "DS1"
#prefix = ""

data_fname = data_dir + exp_name + "/" + exp_name + ".nex"
tree_fname_prefix = res_dir + exp_name + "/" + prefix + "particle_tree"
log_branch_lengh_fname = res_dir + exp_name + "/" + prefix + "samp_log_branch.npy"
log_ll_fname = res_dir + exp_name + "/" + prefix + "log_ll.npy"

processed_trees, processed_trees_newicks = load_particle_trees(n_particles, n_taxa,
                                                               tree_fname_prefix, log_branch_lengh_fname)

# Save tree images
for p in range(n_particles):
    fname = tree_fname_prefix + "_processed_" + str(p) + ".png"
    save_tree_image(fname, processed_trees_newicks[p], n_taxa)

# Compare log-likelihoods
vbpi_nf_log_ll = np.load(log_ll_fname)
data = read_nex_file(data_fname)
compare_log_likelihoods(processed_trees, data, vbpi_nf_log_ll, processed_trees_newicks)
'''