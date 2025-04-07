import REUParsing as rp
import os
from tqdm import tqdm
import pickle
import random
import shutil

root_addr = "D:\\REU Datasets\\Universal Dependencies 2.8.1\\ud-treebanks-v2.8"
attrs = ["Pos + Rel", "Rel", "Unlabeled"]
random.seed(123)
datasizes = {}
for attr in attrs:
    path = root_addr
    diversities = {}
    datasizes[attr] = {}
    for dirname in tqdm(os.listdir(root_addr)):
        path = os.path.join(root_addr, dirname)
        os.mkdir(os.path.join("D:\\REU Datasets\\Universal Dependencies 2.8.1\\diverse-treebanks", attr, dirname))
        for file in os.listdir(path):
            path = os.path.join(root_addr, dirname, file)
            if file.endswith("-train.conllu"):
                trees = rp.load_sentences(path)
                hashes = rp.generate_hashes(trees,
                    edge_attr=("deprel" if not attr == "Unlabeled" else None),
                    node_attr=("upos" if attr == "Pos + Rel" else None))
                diversified_trees = []
                for hash_ in hashes:
                    diversified_trees.append(random.choice(hashes[hash_]))
                datasizes[attr][dirname] = len(diversified_trees)
                with open(os.path.join("D:\\REU Datasets\\Universal Dependencies 2.8.1\\diverse-treebanks", attr, dirname, file), "wt", encoding="utf-8") as f:
                    for tree in diversified_trees:
                        f.write(tree.serialize())
            if file.endswith("-test.conllu") or file.endswith("-dev.conllu"):
                shutil.copyfile(path, os.path.join("D:\\REU Datasets\\Universal Dependencies 2.8.1\\diverse-treebanks", attr, dirname, file))

print(datasizes)