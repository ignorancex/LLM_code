import os
import sys

sys.path.append(os.getcwd())

import logging
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from main import parse_args
from utils import save, load, compute_loglikelihood
from logging_utils import set_logger, close_logger
from phylo_tree import sample_t_jc


def add_additional_arguments(args):
    args.n_trees = 500
    args.n_distorted = 5
    args.distortion_rate = 4
    args.branch_strategy = "jc"
    args.log_filename += "_post_prep.log"


def get_occurrence(model, particle_trees):
    node_cooccurrence_mat = np.zeros((model.n_nodes, model.n_nodes))
    for tree in particle_trees:
        for u, v in tree.edges:
            node_cooccurrence_mat[u, v] += 1
            node_cooccurrence_mat[v, u] = node_cooccurrence_mat[u, v]
    node_cooccurrence_mat /= len(particle_trees)
    return node_cooccurrence_mat


def plot_occurrence(matrix, filename, title=None, axis_labels=None):
    plt.figure(figsize=(6, 6))
    plt.imshow(matrix, interpolation='nearest')
    if title is not None:
        plt.title(title)
    if axis_labels is not None:
        plt.xlabel(axis_labels)
        plt.ylabel(axis_labels)
    plt.colorbar()
    plt.savefig(filename)
    plt.close()


def run(log_W, t_opts, n_trees, branch_strategy):
    # Sample topologies
    particle_trees, particle_log_q_list = model.sample_trees(n_trees=n_trees, log_W=log_W, t_opts=t_opts, phi=None, return_q=True)  # TODO adjust phi later, if csmc will be used

    # Sample branch lengths
    if branch_strategy != 'ml':
        for tree in particle_trees:
            for u, v in tree.edges:
                tree[u][v]["t"] = sample_t_jc(S_ij=model.S_tensor[u, v])
    return particle_trees, list(particle_log_q_list)


def get_likelihoods(model, particle_trees):
    particle_lls = []
    for tree in particle_trees:
        particle_lls.append(compute_loglikelihood(model.phylo.compute_up_messages(tree.copy())))
    return particle_lls


def get_leaves(tree, n_taxa):
    tree_nodes = np.array(tree.nodes())
    return tree_nodes[tree_nodes < n_taxa]


def get_bitmask(model, particle_trees, particle_lls=None):  # TODO might need to debug more for trans-latent, multifurcating trees  # TODO move to bitmask_utils.py later
    """ Inspired by https://dendropy.org/primer/bipartitions.html """
    bitmask_dict = {}

    for i in range(len(particle_trees)):
        tree_bitmask_dict = {}
        tree = particle_trees[i].copy()

        root_list = [model.root]
        while len(root_list) != 0:
            cur_root = root_list[0]
            try:
                for child in tree._node[cur_root]['children']:
                    if child >= model.n_taxa:
                        root_list.append(child)
            except:
                print("Trans-latent node! ", cur_root, tree._node[cur_root])
            root_list = root_list[1:]

            subtrees = [tree.subgraph(c).copy() for c in nx.connected_components(tree)]
            for subtree in subtrees:
                if cur_root in list(subtree.nodes()):
                    leaves_list = get_leaves(subtree, model.n_taxa)
                    # Only consider non-trivial vertices
                    if len(leaves_list) > 1:
                        mask = np.zeros((model.n_taxa), dtype=int)
                        mask[leaves_list] = 1
                        bitmask_str = "".join(map(str, list(mask)))
                        # This if makes sure we don't repeat same string within a tree,
                        # e.g caused by ladderizes trans-latent nodes
                        if bitmask_str not in tree_bitmask_dict:
                            tree_bitmask_dict[bitmask_str] = 1
                            if particle_lls is None:
                                if bitmask_str in bitmask_dict:
                                    bitmask_dict[bitmask_str] += 1
                                else:
                                    bitmask_dict[bitmask_str] = 1
                            else:
                                tree_ll = particle_lls[i]
                                if bitmask_str in bitmask_dict:
                                    bitmask_dict[bitmask_str]["count"] += 1
                                    bitmask_dict[bitmask_str]["ll"] += tree_ll
                                else:
                                    bitmask_dict[bitmask_str] = {}
                                    bitmask_dict[bitmask_str]["count"] = 1
                                    bitmask_dict[bitmask_str]["ll"] = tree_ll
            tree.remove_node(cur_root)

    if particle_lls is not None:
        for bitmask_str in bitmask_dict:
            bitmask_dict[bitmask_str]["ll"] /= bitmask_dict[bitmask_str]["count"]

    return bitmask_dict


def get_bitmask_trees(model, particle_trees, particle_lls, particle_log_q_list):  # TODO might need to debug more for trans-latent, multifurcating trees  # TODO move to bitmask_utils.py later
    """ Inspired by https://dendropy.org/primer/bipartitions.html """
    bitmask_dict = {}

    for i in range(len(particle_trees)):
        tree_bitmask_dict = {'partitions': [], 'll': particle_lls[i], 'log_q': particle_log_q_list[i]}

        tree = particle_trees[i].copy()

        root_list = [model.root]
        while len(root_list) != 0:
            cur_root = root_list[0]
            try:
                for child in tree._node[cur_root]['children']:
                    if child >= model.n_taxa:
                        root_list.append(child)
            except:
                print("Trans-latent node! ", cur_root, tree._node[cur_root])
            root_list = root_list[1:]

            subtrees = [tree.subgraph(c).copy() for c in nx.connected_components(tree)]
            for subtree in subtrees:
                if cur_root in list(subtree.nodes()):
                    leaves_list = get_leaves(subtree, model.n_taxa)
                    # Only consider non-trivial vertices
                    if len(leaves_list) > 1:
                        mask = np.zeros((model.n_taxa), dtype=int)
                        mask[leaves_list] = 1
                        bitmask_str = "".join(map(str, list(mask)))
                        # This if makes sure we don't repeat same string within a tree,
                        # e.g caused by ladderizes trans-latent nodes
                        if bitmask_str not in tree_bitmask_dict['partitions']:
                            part_so_far = tree_bitmask_dict['partitions']
                            part_so_far.append(bitmask_str)
                            tree_bitmask_dict['partitions'] = part_so_far
            tree.remove_node(cur_root)
        bitmask_dict[i] = tree_bitmask_dict
    return bitmask_dict


if __name__ == "__main__":
    print("Hello, world!")

    # Parse arguments
    args = parse_args()
    # Add additional arguments required for post analysis preparation
    add_additional_arguments(args)

    # Set console_level=logging.INFO or console_level=logging.DEBUG
    set_logger(filename=args.log_filename, console_level=logging.INFO)

    if args.model == "vaiphy":  # TODO add other models when needed
        import vaiphy.vaiphy as model

    # Load model
    filename = args.file_prefix + '_object.pkl'
    model = load(filename)
    logging.info("Model is loaded.")
    model.report_edmonds_tree()

    # Create output directory
    out_dir = model.result_path + '/post_analysis_prep'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    logging.info("Results will be saved to: %s" % out_dir)

    # Save expected counts
    fname = out_dir + "/exp_counts.npy"
    np.save(fname, model.S_tensor, allow_pickle=False)

    logging.info("n_trees: %s, n_distorted: %s, distortion_rate: %s, branch_strategy: %s"
                 % (args.n_trees, args.n_distorted, args.distortion_rate, args.branch_strategy))

    # Set seed
    np.random.seed(model.seed)

    # Sample non-distorted trees
    particle_trees, particle_log_q_list = run(log_W=model.log_W_list[-1], t_opts=model.t_opts_list[-1],
                                              n_trees=args.n_trees, branch_strategy=args.branch_strategy)  # Note this is different than model's branch strategy
    particle_lls = get_likelihoods(model, particle_trees)
    logging.info("Particle Log-likelihoods. Mean: %s \tStd: %s \tMax: %s"
                 % (np.mean(particle_lls), np.std(particle_lls), np.max(particle_lls)))

    # Save non-distorted results and plot
    node_cooccurrence_mat = get_occurrence(model, particle_trees)
    fname = out_dir + "/node_cooccurrence_mat.npy"
    np.save(fname, node_cooccurrence_mat, allow_pickle=False)
    filename = out_dir + "/node_cooccurrence_mat.png"
    plot_occurrence(node_cooccurrence_mat, filename, title="Node Cooccurrence", axis_labels="Nodes")

    bitmask_dict_orig = get_bitmask(model, particle_trees, particle_lls=None)
    fname = out_dir + "/bitmask_dict_orig.pkl"
    save(bitmask_dict_orig, fname)
    print("bitmask_dict_orig:\n", bitmask_dict_orig)

    bitmask_dict = get_bitmask(model, particle_trees, particle_lls=particle_lls)
    fname = out_dir + "/bitmask_dict.pkl"
    save(bitmask_dict, fname)
    #print("bitmask_dict:\n", bitmask_dict)

    bitmask_dict_trees = get_bitmask_trees(model, particle_trees, particle_lls, particle_log_q_list)
    fname = out_dir + "/bitmask_trees_dict.pkl"
    save(bitmask_dict_trees, fname)
    #print("bitmask_dict:\n", bitmask_dict_trees)

    # Sample distorted trees
    for distort_idx in range(args.n_distorted):
        np.random.seed(distort_idx)

        logging.info("\tDistortion rate: %s distortion idx: %s" % (args.distortion_rate, distort_idx))
        log_W_temp = model.phylo.compute_edge_weights(model.n_nodes, model.g_list[-1], model.t_opts_list[-1],
                                                      distort_S=True, poi_distortion_rate=args.distortion_rate)

        distorted_particle_trees, distorted_particle_log_q_list = run(log_W=log_W_temp, t_opts=model.t_opts_list[-1],
                                                                      n_trees=args.n_trees, branch_strategy=args.branch_strategy)  # Note this is different than model's branch strategy
        distorted_particle_lls = get_likelihoods(model, distorted_particle_trees)
        logging.info("\tParticle Log-likelihoods. Mean: %s \tStd: %s \tMax: %s"
                     % (np.mean(distorted_particle_lls), np.std(distorted_particle_lls), np.max(distorted_particle_lls)))

        # Save distorted results
        node_cooccurrence_mat = get_occurrence(model, distorted_particle_trees)
        fname = out_dir + "/node_cooccurrence_mat_distorted_lambda_" + str(args.distortion_rate) \
                + "_rep_" + str(distort_idx) + ".npy"
        np.save(fname, node_cooccurrence_mat, allow_pickle=False)

        particle_trees += distorted_particle_trees
        particle_log_q_list += distorted_particle_log_q_list
        particle_lls += distorted_particle_lls

    if args.n_distorted > 0:
        # Save combined results and plot
        node_cooccurrence_mat = get_occurrence(model, particle_trees)
        node_cooccurrence_mat = node_cooccurrence_mat / (args.n_distorted + 1)
        fname = out_dir + "/node_cooccurrence_mat_distorted_lambda_" + str(args.distortion_rate) + "_combined.npy"
        np.save(fname, node_cooccurrence_mat, allow_pickle=False)
        filename = out_dir + "/node_cooccurrence_mat_distorted_lambda_" + str(args.distortion_rate) + "_combined.png"
        title_str = "Node Cooccurrence Combined. Lambda=" + str(args.distortion_rate)
        plot_occurrence(node_cooccurrence_mat, filename, title=title_str, axis_labels="Nodes")

    bitmask_dict_orig = get_bitmask(model, particle_trees, particle_lls=None)
    fname = out_dir + "/bitmask_dict_distorted_lambda_" + str(args.distortion_rate) + "_combined_orig.pkl"
    save(bitmask_dict, fname)
    print("bitmask_dict_orig:\n", bitmask_dict_orig)

    bitmask_dict = get_bitmask(model, particle_trees, particle_lls=particle_lls)
    fname = out_dir + "/bitmask_dict_distorted_lambda_" + str(args.distortion_rate) + "_combined.pkl"
    save(bitmask_dict, fname)
    #print("bitmask_dict:\n", bitmask_dict)

    bitmask_dict_trees = get_bitmask_trees(model, particle_trees, particle_lls, particle_log_q_list)
    fname = out_dir + "/bitmask_trees_dict_distorted_lambda_" + str(args.distortion_rate) + "_combined.pkl"
    save(bitmask_dict_trees, fname)
    #print("bitmask_dict:\n", bitmask_dict_trees)

    close_logger()
