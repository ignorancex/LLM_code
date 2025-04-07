import argparse
import gensim
import json
import numpy as np
import os
import pandas as pd
import torch

from scipy.fftpack import dct
from nltk.tokenize import word_tokenize
from tqdm import tqdm


def transform_sentences(sentences, _model, _k):
    # Only big GloVe is cased, so pre-processing may have to be done
    # _df['tokenize_sent'] = _df['sentText'].apply(word_tokenize)
    # sentences = _df['sentText'].tolist()
    _sent_embs = []
    _dim = 300
    for sent in tqdm(sentences):
        _cur_vec = []
        for tok in sent:
            try:
                _cur_vec.append(_model[tok].astype(np.float64))
            except KeyError:
                pass
        _cur_vec = np.reshape(dct(_cur_vec, n=_k, norm='ortho', axis=0)[:_k, :], (_k * _dim,))
        _sent_embs.append(_cur_vec)
        del _cur_vec
    _sent_tensors = [torch.from_numpy(_j) for _j in _sent_embs]
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
    gensim_path = wd + "/sentence-embeddings/glove/pretrained/gensim/glove_6B_300d_model.txt"
    model = gensim.models.KeyedVectors.load_word2vec_format(gensim_path, binary=False)
    print('... model loaded.')
    for j in range(6):
        train_sent_space = transform_sentences(sentences, model, j+1)
        torch.save(train_sent_space, '{d}_dct_{k}_space.pt'.format(d=args.data, k=j+1))
        print(train_sent_space.shape)
