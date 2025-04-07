import conllu
import networkx as nx
import REUParsing as rp
import os
from tqdm import tqdm
import shutil
import pickle
import sys
import random

def main():
    random.seed(123)
    banks = {"UD_Faroese-FarPaHC", "UD_German-HDT", "UD_Afrikaans-AfriBooms",
            "UD_Danish-DDT", "UD_Icelandic-IcePaHC", "UD_Norwegian-Bokmaal",
            "UD_Swedish-LinES"}
    leakage_attr = ["Unlabeled", "Pos + Rel", "Rel"]
    sizes = [979, 4380, 1019]
    leaky_save_dir = "../../Universal-Dependencies-2.8.1/faroese-leaky-trimmed/"
    nonleaky_save_dir = "../../Universal-Dependencies-2.8.1/faroese-nonleaky-trimmed/"
    for leakage, size in zip(leakage_attr, sizes):
        leaky_dir = "../../Universal-Dependencies-2.8.1/faroese-leaky/" + leakage + "/"
        nonleaky_dir = "../../Universal-Dependencies-2.8.1/faroese-nonleaky/" + leakage + "/"
        print(f"Trimming {leakage} data...")
        # Treebanks will be the same in both leaky and nonleaky directories
        for treebank in tqdm(os.listdir(leaky_dir)):
            if treebank not in banks:
                continue
            # Find the train files
            leakytrees = []
            # nonleakytrees = []
            for file in os.listdir(leaky_dir+treebank):
                if file.endswith("-train.conllu"):
                    leakytrees = rp.load_sentences(leaky_dir+treebank+"/"+file)
                    leakyfile = file
                    break
            # for file in os.listdir(nonleaky_dir+treebank):
            #     if file.endswith("-train.conllu"):
            #         nonleakytrees = rp.load_sentences(nonleaky_dir+treebank+"/"+file)
            #         nonleakyfile = file
            #         break
            
            # Make sure that the files were found
            if not leakytrees:
                continue

            # Don't do anything if size is equal
            # if len(leakytrees) < len(nonleakytrees):
            # os.mkdir(nonleaky_save_dir+leakage+"/"+treebank)
            # # newsize = len(leakytrees)
            # random.shuffle(nonleakytrees)
            # nonleakytrees = nonleakytrees[:size]
            # with open(nonleaky_save_dir+leakage+"/"+treebank+"/"+nonleakyfile, "wt", encoding="utf-8") as nonleaky_save_file:
            #     for tree in nonleakytrees:
            #         nonleaky_save_file.write(tree.serialize())
            # elif len(leakytrees) > len(nonleakytrees):
            os.mkdir(leaky_save_dir+leakage+"/"+treebank)
            # newsize = len(nonleakytrees)
            random.shuffle(leakytrees)
            leakytrees = leakytrees[:size]
            with open(leaky_save_dir+leakage+"/"+treebank+"/"+leakyfile, "wt", encoding="utf-8") as leaky_save_file:
                for tree in leakytrees:
                    leaky_save_file.write(tree.serialize())

            # if newsize < smallest_size[1] and treebank in banks:
            #     smallest_size = (treebank, newsize)


if __name__ == "__main__":
    main()