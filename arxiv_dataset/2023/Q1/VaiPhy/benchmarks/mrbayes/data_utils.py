import os
from Bio import AlignIO
import numpy as np
import re
from Bio.Seq import Seq
from Bio import SeqIO


def remove_char(input_file, output_file, regEX=r"-"):
    #regEX = r"-"
    pattern = re.compile(regEX)
    alignment = SeqIO.parse(input_file, "nexus")
    gap_columns = []
    for record in alignment:
        temp_str = str(record.seq)
        record_len = len(temp_str)
        #print(temp_str)
        res = [i.start() for i in pattern.finditer(temp_str)]
        #print(res)
        #gap_columns.append(res)
        gap_columns +=res

    print(record_len)
    print(len(set(gap_columns)))
    #print(list(range(record_len)))
    #print(list(set(gap_columns)))
    no_gap_columns = np.setdiff1d(list(range(record_len)), list(set(gap_columns)))
    print(len(no_gap_columns))

    alignment = SeqIO.parse(input_file, "nexus")
    with open(output_file, "w") as o:
        records = []
        for record in alignment:
            temp_str = []
            for i in no_gap_columns:
                temp_str.append(str(record.seq[i]))
            #print(record.seq[np.array((2,7))])
            print(''.join(temp_str))
            record.seq = Seq(''.join(temp_str))
            records.append(record)
        SeqIO.write(records, o, "nexus")


# read NEXUS format
def read_nex_file(file_nex):
    alignment = SeqIO.parse(file_nex, "nexus")

    data = []
    for record in alignment:
        data.append(list(record.seq))

    n_leaves = len(data)
    pos = len(data[0])
    return  np.array(data).reshape(n_leaves, pos)


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

    #ali = AlignIO.read(file_temp, "phylip")
    #AlignIO.write(ali, open(file_nex, "w"), "nexus")
    AlignIO.convert(file_temp, "phylip", open(file_nex, "w"), "nexus", molecule_type="DNA")
    os.remove(file_temp)


def save_true_tree(file_path, newick_str, LL):
    f = open(file_path, "a+")
    line = str(LL) + "\t" + newick_str + "\n"
    f.writelines(line)
    f.close()


def read_tree(file_path):
    f = open(file_path, "r")
    lines = f.readlines()
    newick_str = []
    LL = []
    for line in lines:
        LL.append(line.split("\t")[0])
        newick_str.append([line.split("\t")[1]])
    f.close()
    return newick_str, LL


def dataset_generator(dataset_dir, pos):
    dataset_dir = "dataset_" + str(pos)+"pos"
    os.makedirs(dataset_dir, exist_ok=True)
    taxa = [10, 20, 30, 40, 50]
    seed = 123

    np.random.seed(seed)
    tree_path = dataset_dir + "/tree" + ".txt"

    for n_leaves in taxa:
        os.makedirs(dataset_dir, exist_ok=True)

        true_tree = create_tree(n_leaves)
        data = simulate_seq(true_tree, pos)

        # write as .nex files
        file_nex = dataset_dir + "/taxa" + str(n_leaves) + ".nex"
        save_seq_as_nex(file_nex, data)

        # save tree and its LL
        phylo = PhyloTree(data)
        up_table = phylo.compute_up_messages(true_tree)
        true_ll = compute_loglikelihood(up_table)
        newick_str = nx_to_newick(true_tree, 2*n_leaves-3)
        save_true_tree(tree_path, newick_str, true_ll)


if __name__ == '__main__':
    from utils import simulate_seq, compute_loglikelihood
    from tree_utils import create_tree, nx_to_newick
    import numpy as np
    from phylo_tree import PhyloTree

    # test single dataset
    data_path = "dataset_100pos/taxa10.nex"
    dataset_generator(data_path, 100)
    data = read_nex_file(data_path)
    print(data.shape)
    tree_path = "dataset_100pos/tree.txt"
    newick_str, LL = read_tree(tree_path)
    print(LL[0], newick_str[0][0])
