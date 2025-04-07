################################################################################
import sys
import os
import re
import numpy as np
import networkx as nx

# ignore ComplexWarning:  Casting complex values to real discards the imaginary part
import warnings
warnings.simplefilter(action='ignore', category=np.ComplexWarning)
################################################################################
#####--------------------------functions_start-----------------------------#####
def check_folders_existence(folder_path_list):
    """
        given a list of folder path, if not exist, create the path
    """
    for folder_path in folder_path_list:
        if not os.path.exists(folder_path):
        	os.makedirs(folder_path)

#------------------------------------------------------------------------------#
def write_list_to_line_in_file(given_list,file):
    """
        write list into a line in a given file
        line:
        item_1 \t item_2 \t ... \t item__n \n
    """
    line = "\t".join([str(i) for i in given_list])
    file.write(f'{line}\n')

#------------------------------------------------------------------------------#
def save_list_to_file(given_list,file_name):
    """
        write each item into a file as a single line
    """
    f = open(file_name,"w")
    for item in given_list:
        f.write(f'{item}\n')
    f.close()

#------------------------------------------------------------------------------#
def save_dict_to_file(given_dict,file_name):
    """
        write each key-value pair as a line into a file as a single line
    """
    f = open(file_name,"w")
    for k, v in given_dict.items():
        f.write(f'{k}\t{v}\n')
    f.close()

#------------------------------------------------------------------------------#
def save_lists_to_file(given_list_of_lists,file_name):
    """
        write each item into a file as a single line
    """
    f = open(file_name,"w")
    for given_list in given_list_of_lists:
        line = "\t".join([str(i) for i in given_list])
        f.write(f'{line}\n')
    f.close()

#------------------------------------------------------------------------------#
def matrix_to_file(matrix, matrix_file):
    outfile = open(matrix_file, 'w')
    for row in matrix:
        if isinstance(row[0], str):
            line = '\t'.join([i for i in row])
        else:
            line = '\t'.join([str(float(i)) for i in row])
        outfile.write(f'{line}\n')
    outfile.close()

#------------------------------------------------------------------------------#
def array_to_file(array, array_file):
    outfile = open(array_file, 'w')
    lines = '\n'.join([str(float(i)) for i in array])
    outfile.write(f'{lines}')
    outfile.close()

#------------------------------------------------------------------------------#
def file_to_matrix(file):
    matrix = np.loadtxt(file, delimiter='\t')
    return matrix

#------------------------------------------------------------------------------#
def file_to_array(file):
    array = np.loadtxt(file)
    return array

#------------------------------------------------------------------------------#
def get_graph_from_file(network_file, **kwargs):
    """
        generate a graph based on the input file
        The input file is provided by Joerg Menche et al. in their paper's supplementary
        Thus use their function to parse the file and get the graph
        The function returns:
        G: the graph with self loop removed
    """

    defaultKwargs = {'self_link': True}
    kwargs = { **defaultKwargs, **kwargs}

    G = nx.Graph()
    network_file = open(network_file,'r')
    for line in network_file:
        # lines starting with '#' will be ignored
        if line[0]=='#':
            continue
        line_data   = line.strip().split('\t')
        gene1 = line_data[0]
        gene2 = line_data[1]

        G.add_edge(gene1,gene2)

    # remove self links
    if not kwargs['self_link']:
        remove_self_links(G)
    return G

#------------------------------------------------------------------------------#
def remove_self_links(G):
    sl = nx.selfloop_edges(G)
    G.remove_edges_from(sl)

#------------------------------------------------------------------------------#
def analyze_graph(G):
    all_genes_in_network = set(G.nodes())
    print(f'node size: {len(all_genes_in_network)}')
    # get a list of length of the connected commponents
    component_len_list = [len(c) for c in sorted(nx.connected_components(G), key=len, reverse=True)]
    print(f'component len: {component_len_list}')
    print(f'sum of the componenet len: {sum(component_len_list)}')

#------------------------------------------------------------------------------#
def get_gene_idx_dict_from_file(node_file_name):
    f = open(node_file_name, "r")
    gene_idx_dict = {}
    idx = 0
    for line in f:
        node = line.strip()
        gene_idx_dict[node] = idx
        idx += 1
    f.close()
    return gene_idx_dict

#------------------------------------------------------------------------------#
def get_disease_pair_dict(selected_disease_genes_file_path):
    f = open(selected_disease_genes_file_path, "r")
    disease_pair_dict = {}

    header_line = True
    for line in f:
        # skip the headline
        if header_line:
            header_line = False
            continue
        [disease_pair, rr] = line.split("\t")
        disease_pair_dict[disease_pair] = rr
    
    f.close()
    return disease_pair_dict

#------------------------------------------------------------------------------#
def str_to_set(gene_list_str):
    gene_list = re.sub('[^0-9,]+', '', gene_list_str).split(",")
    gene_set = set(gene_list)
    return gene_set

#####--------------------------functions_end-------------------------------#####
