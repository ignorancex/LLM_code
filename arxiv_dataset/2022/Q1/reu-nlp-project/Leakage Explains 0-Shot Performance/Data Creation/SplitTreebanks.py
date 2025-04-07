import conllu
import networkx as nx
import REUParsing as rp
import os
from tqdm import tqdm
import shutil
import pickle
import sys


def main():
    leakage_attr = "Rel" #"Pos + Rel" #"Unlabeled"

    root_addr = "D:\\REU Datasets\\Universal Dependencies 2.8.1\\ud-treebanks-v2.8\\"
    leaky_addr = "D:\\REU Datasets\\Universal Dependencies 2.8.1\\leaky-treebanks\\" + leakage_attr + "\\"
    nonleaky_addr = "D:\\REU Datasets\\Universal Dependencies 2.8.1\\nonleaky-treebanks\\" + leakage_attr + "\\"

    with open(leakage_attr+"\\leaky_train_hashes.dict", "rb") as f:
        leaky_train_hashes = pickle.load(f)

    with open(leakage_attr+"\\nonleaky_train_hashes.dict", "rb") as f:
        nonleaky_train_hashes = pickle.load(f)

    with open(leakage_attr+"\\leaky_test_hashes.dict", "rb") as f:
        leaky_test_hashes = pickle.load(f)

    with open(leakage_attr+"\\nonleaky_test_hashes.dict", "rb") as f:
        nonleaky_test_hashes = pickle.load(f)

    with open(leakage_attr+"\\leaky_dev_hashes.dict", "rb") as f:
        leaky_dev_hashes = pickle.load(f)

    with open(leakage_attr+"\\nonleaky_dev_hashes.dict", "rb") as f:
        nonleaky_dev_hashes = pickle.load(f)

    for dirname in tqdm(os.listdir(root_addr)):
        treebank = dirname
        directory = root_addr + dirname
        if treebank not in os.listdir(leaky_addr):
            os.mkdir(leaky_addr+treebank)
        leaky_dir = leaky_addr+treebank
        if treebank not in os.listdir(nonleaky_addr):
            os.mkdir(nonleaky_addr+treebank)
        nonleaky_dir = nonleaky_addr+treebank
        
        for file in os.listdir(directory):
            file_addr = directory + "\\" + file

            if file.endswith(".conllu"):
                with open(leaky_dir+"\\"+file, "wt", encoding="utf-8") as leaky_file:
                    with open(nonleaky_dir+"\\"+file, "wt", encoding="utf-8") as nonleaky_file:
                        trees = rp.load_sentences(file_addr)
                        for tree in trees:
                            hash_ = rp.get_hash(tree,
                                edge_attr=(None if leakage_attr == "Unlabeled" else "deprel"),
                                node_attr=(None if leakage_attr != "Pos + Rel" else "upos"))
                            if file.endswith("-train.conllu"):
                                if hash_ in leaky_train_hashes[treebank]:
                                    leaky_file.write(tree.serialize())
                                    leaky_file.write("\n\n")
                                if hash_ in nonleaky_train_hashes[treebank]:
                                    nonleaky_file.write(tree.serialize())
                                    nonleaky_file.write("\n\n")
                            if file.endswith("-test.conllu"):
                                if hash_ in leaky_test_hashes[treebank]:
                                    leaky_file.write(tree.serialize())
                                    leaky_file.write("\n\n")
                                if hash_ in nonleaky_test_hashes[treebank]:
                                    nonleaky_file.write(tree.serialize())
                                    nonleaky_file.write("\n\n")
                            if file.endswith("-dev.conllu"):
                                if hash_ in leaky_dev_hashes[treebank]:
                                    leaky_file.write(tree.serialize())
                                    leaky_file.write("\n\n")
                                if hash_ in nonleaky_dev_hashes[treebank]:
                                    nonleaky_file.write(tree.serialize())
                                    nonleaky_file.write("\n\n")

if __name__ == "__main__":
    main()