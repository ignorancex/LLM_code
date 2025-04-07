import REUParsing as rp
import REULeakage as rl
import numpy as np
import pickle
import conllu
import networkx as nx
from tqdm import tqdm
from matplotlib import pyplot as plt
from scipy.spatial import distance
import os
import sys

MIN_TREES = 1

# Creates sparse vectors for every treebank in the train and test sets
def prepare_vectors(language_hashes_train, language_hashes_test):
    # Compute a combined dictionary
    language_hashes = {}
    for language in language_hashes_train:
        if language not in language_hashes:
            language_hashes[language] = {}
        for hash_ in language_hashes_train[language]:
            if hash_ not in language_hashes[language]:
                language_hashes[language][hash_] = language_hashes_train[language][hash_]
            else:
                language_hashes[language][hash_] += language_hashes_train[language][hash_]
    for language in language_hashes_test:
        if language not in language_hashes:
            language_hashes[language] = {}
        for hash_ in language_hashes_test[language]:
            if hash_ not in language_hashes[language]:
                language_hashes[language][hash_] = language_hashes_test[language][hash_]
            else:
                language_hashes[language][hash_] += language_hashes_test[language][hash_]

    # Compute the counts and remove any treebanks with fewer than MIN_TREES
    train_counts = rl.language_counts(language_hashes_train)
    for lang in train_counts:
        if train_counts[lang] < MIN_TREES:
            del language_hashes_train[lang]
    train_counts = {}
    for lang in language_hashes_train:
        train_counts[lang] = 0
        for hash_ in language_hashes_train[lang]:
            train_counts[lang] += language_hashes_train[lang][hash_]
    test_counts = rl.language_counts(language_hashes_test)
    for lang in test_counts:
        if test_counts[lang] < MIN_TREES:
            del language_hashes_test[lang]
    test_counts = {}
    for lang in language_hashes_test:
        test_counts[lang] = 0
        for hash_ in language_hashes_test[lang]:
            test_counts[lang] += language_hashes_test[lang][hash_]
            
    all_counts = rl.language_counts(language_hashes)
    for lang in all_counts:
        if all_counts[lang] < MIN_TREES:
            del language_hashes[lang]
    all_counts = {}
    for lang in language_hashes:
        all_counts[lang] = 0
        for hash_ in language_hashes[lang]:
            all_counts[lang] += language_hashes[lang][hash_]

    # Compute the relative hash counts
    relative_train = rl.relative_hashes(language_hashes_train, train_counts)
    relative_test = rl.relative_hashes(language_hashes_test, test_counts)
    relative_all = rl.relative_hashes(language_hashes, all_counts)

    # Create a vocab of hashes we have seen
    hash_vocab = rl.generate_hash_vocab(relative_all)

    # Generate vectors
    print("Generating 'train' vectors...")
    train_vectors = rl.language_vectors(relative_train, hash_vocab)
    print("Generating 'test' vectors...")
    test_vectors = rl.language_vectors(relative_test, hash_vocab)

    return train_vectors, test_vectors

# The leakages for each treebank from its test set into its own train set
def per_treebank_leakage(train_vectors, test_vectors):
    per_treebank_leakages = {}
    for treebank in tqdm(test_vectors):
        if treebank in train_vectors:
            per_treebank_leakages[treebank] = rl.leakage(test_vectors[treebank], {treebank:train_vectors[treebank]})
        else:
            per_treebank_leakages[treebank] = -1
    return per_treebank_leakages

# The leakages for each treebank from its test set into all other training sets
def cross_treebank_leakage(train_vectors, test_vectors):
    cross_treebank_leakages = {}
    for treebank in tqdm(test_vectors):
        group = {bank:train_vectors[bank] for bank in train_vectors if bank != treebank}
        cross_treebank_leakages[treebank] = rl.leakage(test_vectors[treebank], group)
    return cross_treebank_leakages

# The leakages for each treebank from its test set into all other training sets (on an individual basis)
def inter_treebank_leakage(train_vectors, test_vectors):
    inter_treebank_leakages = {}
    for testtreebank in tqdm(test_vectors):
        inter_treebank_leakages[testtreebank] = {}
        for traintreebank in train_vectors:
            inter_treebank_leakages[testtreebank][traintreebank] = rl.leakage(test_vectors[testtreebank], {traintreebank:train_vectors[traintreebank]})
    return inter_treebank_leakages

def main():
    if len(sys.argv) < 4:
        print("Usage: [train hashes] [test hashes] [options...]")
        print("Options:" +
              "\n\t-p [output file]: per-treebank leakages" +
              "\n\t-c [output file]: cross-treebank leakages" +
              "\n\t-i [output file]: inter-treebank leakages")
        exit()
    # Load the hash dictionaries
    with open(sys.argv[1], "rb") as lf:
        language_hashes_train = pickle.load(lf)
    with open(sys.argv[2], "rb") as lf:
        language_hashes_test = pickle.load(lf)

    train_vectors, test_vectors = prepare_vectors(language_hashes_train, language_hashes_test)
    
    options = sys.argv[3:]

    if "-p" in options:
        print("Computing per-treebank leakages...")
        per_treebank_leakages = per_treebank_leakage(train_vectors, test_vectors)

        with open(sys.argv[sys.argv.index("-p") + 1], "wb") as f:
            pickle.dump(per_treebank_leakages, f)

    if "-c" in options:
        print("Computing cross-treebank leakages...")
        cross_treebank_leakages = cross_treebank_leakage(train_vectors, test_vectors)

        with open(sys.argv[sys.argv.index("-c") + 1], "wb") as f:
            pickle.dump(cross_treebank_leakages, f)

    if "-i" in options:
        print("Computing inter-treebank leakages...")
        inter_treebank_leakages = inter_treebank_leakage(train_vectors, test_vectors)

        with open(sys.argv[sys.argv.index("-i") + 1], "wb") as f:
            pickle.dump(inter_treebank_leakages, f)
if __name__ == "__main__":
    main()