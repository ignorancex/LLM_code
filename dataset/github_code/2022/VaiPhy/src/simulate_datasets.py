import os
import sys
sys.path.append(os.getcwd())

import numpy as np
from Bio import AlignIO
from phylo_tree import PhyloTree
from data_utils import save_seq_as_nex
from utils import simulate_seq, compute_loglikelihood
from tree_utils import create_tree, nx_to_newick, save_tree_image

print("Hello, world!")

# Some parameters
data_path = "../data/"
if not os.path.exists(data_path):
    os.makedirs(data_path)

for n_pos in [300, 1000]:
    for data_seed in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        for n_leaves in [10, 20, 40]:
            print(f"\nSimulating Taxa = {n_leaves}, \t Pos = {n_pos}, \t Data_seed = {data_seed}...")

            dataset_name = "data_seed_" + str(data_seed) + "_taxa_" + str(n_leaves) + "_pos_" + str(n_pos)

            output_dir = data_path + dataset_name + "/"
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            # Generate Data
            np.random.seed(data_seed)
            true_tree = create_tree(n_leaves)
            data = simulate_seq(true_tree, n_pos)

            # Compute likelihood with true tree
            phylo = PhyloTree(data)
            up_table = phylo.compute_up_messages(true_tree)
            true_ll = compute_loglikelihood(up_table)
            print("\tLog-Likelihood: ", true_ll)

            # Save Newick
            fname = output_dir + dataset_name + ".nw"
            true_tree_newick = nx_to_newick(true_tree, 2 * n_leaves - 3)
            with open(fname, 'w') as f:
                f.write(true_tree_newick)
            print("\tNewick: ", fname)

            # Save tree image
            fname = output_dir + dataset_name + ".png"
            save_tree_image(fname, true_tree_newick, n_taxa=n_leaves)
            print("\tPng: ", fname)

            # Save Nexus
            fname = output_dir + dataset_name + ".nex"
            save_seq_as_nex(fname, data)
            print("\tNexus: ", fname)

            # Save Fasta
            fname2 = output_dir + dataset_name + ".fasta"
            AlignIO.convert(fname, "nexus", open(fname2, "w"), "fasta")
            print("\tFasta: ", fname2)

            # Save Phylip
            fname3 = output_dir + dataset_name + ".phylip"
            AlignIO.convert(fname, "nexus", open(fname3, "w"), "phylip")
            print("\tPhylip: ", fname3)
