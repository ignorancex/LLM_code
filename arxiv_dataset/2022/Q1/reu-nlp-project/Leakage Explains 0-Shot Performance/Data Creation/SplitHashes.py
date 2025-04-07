# Split training and testing hashes between leaky and non-leaky
import sys
import pickle
import copy
from tqdm import tqdm

def main():
    if len(sys.argv) < 10:
        print("Usage: [train hashes] [test hashes] [dev hashes] [leaky train] [non-leaky train] [leaky test] [non-leaky test] [leaky dev] [non-leaky dev]")
        exit()

    # Load hashes
    with open(sys.argv[1], "rb") as f:
        train_hashes = pickle.load(f)
    with open(sys.argv[2], "rb") as f:
        test_hashes = pickle.load(f)
    with open(sys.argv[3], "rb") as f:
        dev_hashes = pickle.load(f)

    # Split train hashes
    leaky_train = {}
    nonleaky_train = {}
    print("Splitting train hashes...")
    for treebank in tqdm(train_hashes):
        leaky_train[treebank] = {}
        nonleaky_train[treebank] = {}
        if treebank in test_hashes:
            for hash_ in train_hashes[treebank]:
                if hash_ in test_hashes[treebank]:
                    leaky_train[treebank][hash_] = train_hashes[treebank][hash_]
                else:
                    nonleaky_train[treebank][hash_] = train_hashes[treebank][hash_]
        else:
            nonleaky_train[treebank] = copy.deepcopy(train_hashes[treebank])

    # Split test hashes
    leaky_test = {}
    nonleaky_test = {}
    print("Splitting test hashes...")
    for treebank in tqdm(test_hashes):
        leaky_test[treebank] = {}
        nonleaky_test[treebank] = {}
        if treebank in train_hashes:
            for hash_ in test_hashes[treebank]:
                if hash_ in train_hashes[treebank]:
                    leaky_test[treebank][hash_] = test_hashes[treebank][hash_]
                else:
                    nonleaky_test[treebank][hash_] = test_hashes[treebank][hash_]
        else:
            nonleaky_test[treebank] = copy.deepcopy(test_hashes[treebank])

    # Split dev hashes
    leaky_dev = {}
    nonleaky_dev = {}
    print("Splitting dev hashes...")
    for treebank in tqdm(dev_hashes):
        leaky_dev[treebank] = {}
        nonleaky_dev[treebank] = {}
        if treebank in train_hashes:
            for hash_ in dev_hashes[treebank]:
                if hash_ in train_hashes[treebank]:
                    leaky_dev[treebank][hash_] = dev_hashes[treebank][hash_]
                else:
                    nonleaky_dev[treebank][hash_] = dev_hashes[treebank][hash_]
        else:
            nonleaky_dev[treebank] = copy.deepcopy(dev_hashes[treebank])


    # Save split hash dicts
    with open(sys.argv[4], "wb") as f:
        pickle.dump(leaky_train, f)
    with open(sys.argv[5], "wb") as f:
        pickle.dump(nonleaky_train, f)
    with open(sys.argv[6], "wb") as f:
        pickle.dump(leaky_test, f)
    with open(sys.argv[7], "wb") as f:
        pickle.dump(nonleaky_test, f)
    with open(sys.argv[8], "wb") as f:
        pickle.dump(leaky_dev, f)
    with open(sys.argv[9], "wb") as f:
        pickle.dump(nonleaky_dev, f)

if __name__ == "__main__":
    main()