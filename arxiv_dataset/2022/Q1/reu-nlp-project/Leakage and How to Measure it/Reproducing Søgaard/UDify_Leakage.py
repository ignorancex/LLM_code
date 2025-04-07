import ComputeLeakage as cl
import pickle
import sys

def main():
    if len(sys.argv) < 4:
        print("Usage: [train hashes] [test hashes] [out dict]")
        exit()
    # Load the hash dictionaries
    with open(sys.argv[1], "rb") as lf:
        treebank_hashes_train = pickle.load(lf)
    with open(sys.argv[2], "rb") as lf:
        treebank_hashes_test = pickle.load(lf)

    # with open("UDify-Scores.tsv", "rt") as udify:
    with open("UDPipe-Scores.tsv", "rt") as udify:
        lines = [l.strip().split('\t') for l in udify.readlines()[1:]]

    train_hashes = {}
    test_hashes = {}

    # Populate the hash dictionaries based on what was training and testing for UDify
    for line in lines:
        treebank = line[0]
        if treebank in treebank_hashes_test:
            test_hashes[treebank] = treebank_hashes_test[treebank]
        else:
            print(treebank, "not in available test data...")
        if line[-1] != 0:
            # Has train data
            if treebank in treebank_hashes_train:
                train_hashes[treebank] = treebank_hashes_train[treebank]
            else:
                print(treebank, "not in available train data...")

    # Compute the vectors
    train_vectors, test_vectors = cl.prepare_vectors(train_hashes, test_hashes)

    # Calculate the leakage
    print("Calculating the leakage...")
    leakages = cl.cross_treebank_leakage(train_vectors, test_vectors)
    print(*[f"{x}:{leakages[x]}" for x in leakages], sep="\n")
    with open(sys.argv[3], "wb") as f:
        pickle.dump(leakages, f)

if __name__ == "__main__":
    main()