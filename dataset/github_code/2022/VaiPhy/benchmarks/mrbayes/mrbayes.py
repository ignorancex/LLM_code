######################################################################################
#
# Authors :
#
#           Email :
#
# mrbayes: MrBayes utility functions.
#####################################################################################


import sys
sys.path.append('../../')
import os
import argparse
from subprocess import call
from Bio import AlignIO
import numpy as np
import dendropy
import networkx as nx
from data_utils import read_nex_file
from tree_utils import nx_to_newick, newick2nx


# to convert alignment from phylip to NEXUS format
def phylip_to_nex(file, file_nex):
    AlignIO.convert(file, "phylip", open(file_nex, "w"), "nexus", molecule_type="DNA")


def run_mrbayes(file_nex, mrbayes_dir, generations=10000, out_file="output"):
    ali = AlignIO.read(file_nex, "nexus")

    if not os.path.exists(mrbayes_dir):
        os.makedirs(mrbayes_dir)

    # change directory to mrbayes working directory
    os.chdir(mrbayes_dir)
    target_file = os.path.basename(file_nex)

    mrmodel_block = "BEGIN MRBAYES; \n " +\
    "set autoclose=yes nowarn=yes Seed=123 Swapseed=123; \n" +\
    "lset nst=1; \n " +\
    "prset statefreqpr=fixed(equal); \n" +\
    "mcmc ngen = %s nruns=10 nchains=4 printfreq=1000 samplefreq=100 savebrlens=yes filename=%s; \n " % (generations, out_file) +\
    "sumt relburnin = yes burninfrac = 0.25  conformat=simple; \n" + \
    "sump relburnin = yes burninfrac = 0.25; \n" + \
    "sump; \n" + \
    "sumt; \n" + \
    "END;"

    # create target file
    AlignIO.write(ali, open(target_file, "w"), "nexus")

    # to add empty model block to nex.temp for checking alignment
    file_nex_data = open(target_file, "a")
    file_nex_data.write(mrmodel_block)
    file_nex_data.close()

    call("mb  %s" % target_file, shell=True)


def run_mrbayes_ss(file_nex, mrbayes_dir, generations=10000, out_file="output"):
    ali = AlignIO.read(file_nex, "nexus")

    if not os.path.exists(mrbayes_dir):
        os.makedirs(mrbayes_dir)

    # change directory to mrbayes working directory
    os.chdir(mrbayes_dir)
    target_file = os.path.basename(file_nex)

    mrmodel_block = "BEGIN MRBAYES; \n " +\
    "set autoclose=yes nowarn=yes Seed=123 Swapseed=123; \n" +\
    "lset nst=1; \n " +\
    "prset statefreqpr=fixed(equal); \n" +\
    "ss ngen = %s nruns=10 nchains=4 printfreq=1000 samplefreq=100 savebrlens=yes filename=%s; \n " % (generations, out_file) + \
    "sump nruns=10 filename=%s; \n" % (out_file) + \
    "sumt nruns=10 contype='Halfcompat' filename=%s; \n" % (out_file) \
    "END;"

    # create target file
    AlignIO.write(ali, open(target_file, "w"), "nexus")

    # to add empty model block to nex.temp for checking alignment
    file_nex_data = open(target_file, "a")
    file_nex_data.write(mrmodel_block)
    file_nex_data.close()

    call("mb  %s" % target_file, shell=True)


# change newick from mrbayes, it starts from 1,2,3...
# we need it from : 0,1,2...
def change_newick_str(newick_str, n_leaves, scale=0.1):
    """ Converts a newick string to Networkx graph. Newick -> Dendropy Tree -> NetworkX graph """
    # Create Dendropy Tree object.
    dendro_tree = dendropy.Tree.get_from_string(newick_str, "newick")

    # Add taxa to internal nodes and convert leaf taxa to integers.
    root_visited = False
    n_nodes = 2 * n_leaves - 2
    node_idx = n_leaves
    for node in dendro_tree.preorder_node_iter():
        # Root
        if not root_visited:
            node.taxon = 2 * n_leaves - 3
            root_visited = True
        else:
            # Internal node
            if node.is_internal():
                node.taxon = node_idx
                node_idx = node_idx + 1
            # Leaf
            else:
                node.taxon = int(str(node.taxon)[1:-1])-1 # Dendropy's leaf taxa has the form: 'name'. We take substring.

    # Convert Dendropy Tree to Networkx graph.
    tree = nx.Graph()
    node_names = [n for n in range(n_nodes)]
    leaves = node_names[:n_leaves]

    # Add nodes to the graph
    for node in leaves:
        tree.add_node(node, type='leaf')
    for node in node_names[n_leaves:-1]:
        tree.add_node(node, type='internal')
    tree.add_node(2 * n_leaves - 3, type='root')

    # Add edges to the graph
    for node in dendro_tree.preorder_node_iter():
        if node.is_internal():
            children = []
            for child_node in node.child_nodes():
                if child_node.edge_length is not None:
                    t = child_node.edge_length
                else:
                    t = np.random.exponential(scale)

                tree.add_edge(node.taxon, child_node.taxon, t=t)
                tree.add_node(child_node.taxon, parent=node.taxon)

                children.append(child_node.taxon)
            tree.add_node(node.taxon, children=children)

    return nx_to_newick(tree, 2 * n_leaves - 3)


# read LL and tree from output file
def read_mrbayes_output(ll_file, tree_file, n_leaves):
    f = open(ll_file, "r")
    lines = f.readlines()
    lines = lines[2:]
    result = []

    for x in lines:
        result.append(x.split('\t')[1])
    f.close()
    result = np.array(result).astype(float)
    max_index = np.argmax(result)

    # get corresponding tree
    f = open(tree_file, "r")
    lines = f.readlines()
    tree = lines[5+n_leaves+max_index]
    tree = tree.split("[&U]")[1]
    f.close()

    return result[max_index], tree, result


def compute_LL(majority_tree, n_leaves, phylo=None):
    nxTree = newick2nx(majority_tree, n_leaves)
    up_table = phylo.compute_up_messages(nxTree)
    cons_tree_ll = phylo.compute_loglikelihood(up_table)
    print("LL", cons_tree_ll)
    return cons_tree_ll


def call_mrbayes(file_path, mrbayes_dir):
    # threads = 1
    generations = 10000000
    # to be deleted
    #generations = 50000
    out_file = "output_ss"

    run_mrbayes_ss(file_path, mrbayes_dir, generations=generations, out_file=out_file)

    ll_max, mb_tree, _ = read_mrbayes_output(out_file + ".run2.p", out_file + ".run2.t", n_leaves)

    return ll_max


def summarize_LL(mrbayes_dir, nruns=10):
    print("Summarizing MrBayes results for nruns: ", nruns)

    #os.chdir(mrbayes_dir)
    out_file = "output_ss"

    all_ll_list = []
    all_ll_list_burnin_0_25 = []
    all_ll_list_burnin_0_50 = []

    for i in range(1, nruns + 1):
        try:
            ll_max, mb_tree, ll_list = read_mrbayes_output(out_file + ".run" + str(i) + ".p",
                                                           out_file + ".run" + str(i) + ".t", n_leaves)
            # No burnin
            all_ll_list.extend(ll_list)
            # First 0.25 burnin
            all_ll_list_burnin_0_25.extend(ll_list[int(np.floor(0.25 * len(ll_list))):])
            # First 0.50 burnin
            all_ll_list_burnin_0_50.extend(ll_list[int(np.floor(0.50 * len(ll_list))):])

            print("\tRun ", i, "\tMax LL: ", ll_max)
        except:
            print("\tWarning! Cannot read Run ", i)

    print("Without BurnIn: \tMean: ", np.mean(all_ll_list), "\tStd: ", np.std(all_ll_list))
    print("With BurnIn (0.25): \tMean: ", np.mean(all_ll_list_burnin_0_25), "\tStd: ", np.std(all_ll_list_burnin_0_25))
    print("With BurnIn (0.50): \tMean: ", np.mean(all_ll_list_burnin_0_50), "\tStd: ", np.std(all_ll_list_burnin_0_50))


if __name__ == "__main__":
    print("Hello, world!")

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, help=' DS1 | DS2 | DS3 | data_seed_1_taxa_10_pos_300 ')
    args = parser.parse_args()

    nex_filename = "../../data/" + args.dataset + "/" + args.dataset + ".nex"
    mrbayes_dir = "results/" + args.dataset

    data = read_nex_file(nex_filename)
    n_leaves, pos = data.shape
    print("n_leaves: ", n_leaves, " pos: ", pos)

    ll_mrbayes = call_mrbayes(nex_filename, mrbayes_dir)
    print("MrBayes Max LogLikelihood of Run 2: ", ll_mrbayes)

    summarize_LL(mrbayes_dir)

