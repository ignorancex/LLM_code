import argparse
import gensim
import json
import numpy as np
import os
import torch

from nltk.tokenize import word_tokenize
from tqdm import tqdm


def transform_sentences(_sentences, _model):
    # Only big GloVe is cased, so preprocessing may have to be done
    # Given low MRR I suspect either
    # a) averaging is not the way to go
    # b) missing a lot of token lookups
    _sentences = [x.lower() for x in _sentences]
    _sentences = [word_tokenize(x) for x in _sentences]  # 1097239 oov tokens
    _sentence_lengths = np.array([len(x) for x in _sentences])
    #print(np.max(_sentence_lengths))
    #print(_sentence_lengths.argmax(axis=0))
    #print(np.min(_sentence_lengths))
    #print(_sentence_lengths.argmin(axis=0))

    _sent_embs = []
    # _oov = {}
    _oov = 0
    for sent in tqdm(_sentences):
        _cur_vec = np.zeros(300).astype(np.float64)
        for tok in sent:
            try:
                _cur_vec += _model[tok].astype(np.float64)
            except KeyError:
                #try:
                #    _oov[tok] += 1
                #except KeyError:
                #    _oov[tok] = 1
                _oov += 1
                pass
        _cur_vec = _cur_vec / float(len(sent))
        _sent_embs.append(_cur_vec)
    _sent_tensors = [torch.from_numpy(j) for j in _sent_embs]
    print(_oov)
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
    gensim_path = os.path.join(wd, "sentence-embeddings/glove/pretrained/gensim/glove_840B_300d_model.txt")
    model = gensim.models.KeyedVectors.load_word2vec_format(gensim_path, binary=False)
    print('... model loaded.')
    sent_space = transform_sentences(sentences, model)
    torch.save(sent_space, '{d}_glove_space.pt'.format(d=args.data))
    print(sent_space.shape)
