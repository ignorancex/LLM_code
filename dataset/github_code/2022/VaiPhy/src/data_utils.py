import os
import sys
sys.path.append(os.getcwd())

import os
import numpy as np
from Bio import AlignIO, SeqIO
from utils import alphabet_size, nuc_vec


# read NEXUS format
def read_nex_file(file_nex):
    alignment = SeqIO.parse(file_nex, "nexus")

    data = []
    for record in alignment:
        data.append(list(record.seq))

    n_leaves = len(data)
    pos = len(data[0])
    return np.array(data).reshape(n_leaves, pos)


def convert_data_str_to_onehot(data):
    n_leaves = len(data)
    n_sites = len(data[0])
    data_onehot = np.ones((n_leaves, n_sites, alphabet_size))
    for i in range(n_leaves):
        data_onehot[i] = np.array([nuc_vec[c] for c in data[i]])
    return data_onehot


def save_seq_as_nex(file_nex, simulated_data):
    cwd = os.getcwd()
    file_temp = cwd + "/temp.phylip"
    f = open(file_temp, "w")
    l, w = simulated_data.shape
    line = "\t" + str(l) + "\t" + str(w) + "\n"
    f.writelines(line)
    for i, line in enumerate(simulated_data):
        l = ' '.join(line)
        l = l.replace(" ", "")
        l = "taxon" + str(i) + "\t\t\t\t" + l + "\n"
        f.writelines(l)
    f.close()

    AlignIO.convert(file_temp, "phylip", open(file_nex, "w"), "nexus", molecule_type="DNA")
    os.remove(file_temp)


def change_taxon_names(input_fname, input_ftype, output_fname, output_ftype, prefix="taxon"):
    """ Example ftypes: "phylip", "nexus" """
    ali_file = AlignIO.read(input_fname, input_ftype)
    idx = 0
    for ali in ali_file:
        if prefix in ali.id:
            break
        print("Renaming: %s%d %s " % (prefix, idx, ali.id))
        ali.id = prefix + str(idx)
        idx += 1
    AlignIO.write(ali_file, open(output_fname, "w"), output_ftype)


def form_dataset_from_strings(genome_strings):
    data_list = []
    n_taxa = len(genome_strings)
    n_sites = len(genome_strings[0])
    for i in range(n_taxa):
        temp_list = []
        for j in range(n_sites):
            temp_list.append(genome_strings[i][j])

        data_list.append(temp_list)

    data = np.array(data_list).reshape(n_taxa, n_sites)
    taxa = ['S' + str(i) for i in range(data.shape[0])]
    datadict = {'taxa': taxa,
                'genome': data}
    return datadict
