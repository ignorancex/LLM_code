import argparse
import gensim
import json
import numpy as np
import os
import torch

from nltk.tokenize import word_tokenize
from tqdm import tqdm


def transform_sentences(_sentences):
    _sentences = [x.lower() for x in _sentences]
    _sent_tensors = torch.randn(len(_sentences), 300)
    return _sent_tensors


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="build_glove_space",
                                     description="Builds sentence embeddings from the GloVe model")
    parser.add_argument('-s', '--set', required=True, type=str,
                        help='The dataset name', dest='data')
    args = parser.parse_args()
    wd = os.path.normpath(os.getcwd() + os.sep + os.pardir + os.sep + os.pardir)
    if args.data == 'riedel':
        sentence_path = os.path.join(wd, "data", "RESIDE", "riedel_data", "sentence_mapper.json")
        with open(sentence_path, 'rb') as f:
            sentences = json.load(f)
    elif args.data == 'gids':
        sentence_path = os.path.join(wd, "data", "GIDS", "gids_data", "sentence_mapper.json")
        with open(sentence_path, 'rb') as f:
            sentences = json.load(f)
    sent_space = transform_sentences(sentences)
    torch.save(sent_space, '{d}_random_space.pt'.format(d=args.data))
    print(sent_space.shape)
