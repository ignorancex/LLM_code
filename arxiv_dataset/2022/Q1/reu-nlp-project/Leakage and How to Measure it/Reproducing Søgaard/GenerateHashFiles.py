import conllu
import networkx as nx
import REUParsing as rp
import os
from tqdm import tqdm
import shutil
import pickle
import sys

root_addr = "D:\\REU Datasets\\Universal Dependencies 2.8.1\\ud-treebanks-v2.8\\"

if len(sys.argv) < 5:
    print("Usage: ['train', 'test', or 'dev'] [dict out file] [edge_attr] [node_attr]")
    exit()

'''
for dirname in tqdm(os.listdir(root_addr)):
    lang = "_".join(dirname.split("-")[0].split("_")[1:])
    directory = root_addr + dirname
    for filename in os.listdir(directory):
        if filename.endswith("-train.conllu"):
            hash_filename = filename.split(".")[0] + "-parses.txt"
            file_addr = directory + "\\" + filename
            hash_file_addr = directory + "\\" + hash_filename
            trees = rp.load_sentences(file_addr)
            with open(hash_file_addr, 'wt') as hash_file:
                for tree in trees:
                    hash_ = nx.weisfeiler_lehman_graph_hash(rp.ud_2_graph(tree),edge_attr='deprel')
                    hash_file.write(str(hash_)+"\n")
            shutil.copy2(hash_file_addr, '.\\Hash-Files\\')

'''
language_hashes = {}
for dirname in tqdm(os.listdir(root_addr)):
    treebank = dirname
    if treebank not in language_hashes:
        language_hashes[treebank] = {}
    directory = root_addr + dirname
    for file in os.listdir(directory):
        if file.endswith(f"-{sys.argv[1]}.conllu"):
            file_addr = directory + "\\" + file
            trees = rp.load_sentences(file_addr)
            hashes = rp.generate_hashes(trees,
                                        edge_attr=sys.argv[3] if sys.argv[3] != str(-1) else None,
                                        node_attr=sys.argv[4] if sys.argv[4] != str(-1) else None)
            for hash_ in hashes:
                if hash_ not in language_hashes[treebank]:
                    language_hashes[treebank][hash_] = 0
                language_hashes[treebank][hash_] += len(hashes[hash_])


with open(sys.argv[2], "wb") as f:
    pickle.dump(language_hashes, f)