import argparse
import gensim
import json
import numpy as np
import os
import torch

from nltk.tokenize import word_tokenize
from tqdm import tqdm

from gem import SentenceEmbedder
from embeddings import get_embedding_matrix


def transform_sentences(_sentences, _model_path):
    _sentences = [x.lower() for x in _sentences]
    embedding_matrix, vocab = get_embedding_matrix(_model_path) #'glove.6B.300d.txt')
    embedder = SentenceEmbedder(sentences, embedding_matrix, vocab)
    _sent_embs = embedder.gem(window_size=3, sigma_power=3)
    _sent_tensors = [torch.from_numpy(j) for j in _sent_embs]
    return torch.stack(_sent_tensors)


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
    print('Loading GloVe model...')
    model_path = os.path.join(wd, "sentence-embeddings/gem/pretrained/glove.6B.300d.txt")
    sent_space = transform_sentences(sentences, model_path)
    torch.save(sent_space, '{d}_gem_space.pt'.format(d=args.data))
    print(sent_space.shape)
