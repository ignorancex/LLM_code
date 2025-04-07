import argparse
import gensim
import json
import numpy as np
import os
import torch

from nltk.tokenize import word_tokenize
from tqdm import tqdm

from models import InferSent


def transform_sentences(_sentences, _model, _pretrained):
    _model.cuda()
    if _pretrained == 'ft':
        W2V_PATH = 'fastText/crawl-300d-2M.vec'
    elif _pretrained == 'glove':
        W2V_PATH = 'GloVe/glove.840B.300d.txt'
    _model.set_w2v_path(W2V_PATH)
    _model.build_vocab(_sentences, tokenize=True)
    embeddings = infersent.encode(sentences, tokenize=True, verbose=True)
    _sent_tensors = [torch.from_numpy(j) for j in embeddings]
    return torch.stack(_sent_tensors)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="build_glove_space",
                                     description="Builds sentence embeddings from the GloVe model")
    parser.add_argument('-s', '--set', required=True, type=str,
                        help='The dataset name', dest='data')
    parser.add_argument('-v', '--version', required=True, type=int,
                        help='The InferSent model version', dest='v')
    parser.add_argument('-p', '--pretrained', required=True, type=str,
                        help='The pretrained vector set', dest='p')
    args = parser.parse_args()
    wd = os.path.normpath(os.getcwd() + os.sep + os.pardir + os.sep + os.pardir + os.sep + os.pardir)
    if args.data == 'riedel':
        sentence_path = os.path.join(wd, "data", "RESIDE", "riedel_data", "sentence_mapper.json")
        with open(sentence_path, 'rb') as f:
            sentences = json.load(f)
    elif args.data == 'gids':
        sentence_path = os.path.join(wd, "data", "GIDS", "gids_data", "sentence_mapper.json")
        with open(sentence_path, 'rb') as f:
            sentences = json.load(f)

    print('Loading InferSent model...')
    MODEL_PATH = 'encoder/infersent{n}.pkl'.format(n=args.v)
    params_model = {'bsize': 256, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                    'pool_type': 'max', 'dpout_model': 0.0, 'version': args.v}
    infersent = InferSent(params_model)
    infersent.load_state_dict(torch.load(MODEL_PATH))
    print('Building sentence representations')
    sent_space = transform_sentences(sentences, infersent, args.p)
    torch.save(sent_space, '{d}_infersent{v}{p}_space.pt'.format(d=args.data,
                                                                 v=args.v,
                                                                 p=args.p))
    print(sent_space.shape)
