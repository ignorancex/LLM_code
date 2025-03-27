import argparse, numpy as np


adjs = ["white", "black", "red", "blue", "purple", "orange", "yellow", "green", "hot", "ice", "beautiful", "dirty", "golden", "muddy", "heavy", "dark", "light"]


nouns = ["aircraft", "album", "ant", "battery", "bicyle", "bridge", "car" "camera", "cherry", "dragon", "fish", "house", "mobilephone", "network", "opera", "oven", "photograph", "playroom", "puzzle", "rabbit", "radish", "sand", "softdrink", "sunflower", "tape", "tower", "water", "watch"]


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--num", type=int, default=None)

    return parser.parse_args()


def main(args):
    idx_adjs = np.random.permutation(np.arange(len(adjs)))
    idx_nouns = np.random.permutation(np.arange(len(nouns)))

    sentences = []
    for i in range(args.num):
        idx1 = idx_adjs[i]
        idx2 = idx_nouns[i]

        sen = "{} {}".format(adjs[idx1], nouns[idx2])
        sentences.append(sen)

    for sen in sentences:
        print("'{}'".format(sen))


if __name__ == "__main__":
    args = get_args()
    main(args)
