'''
THEANO_FLAGS='floatX=float32,device=cuda0,dnn.enabled=False,exception_verbosity=high' python build_skipthought_space.py -s riedel
'''
import argparse
import json
import os
import pandas as pd
import torch

import skipthoughts


def transform_sentences(sentences):
    """
    Builds sentence embeddings using the Skip-thoughts model.

    :param _df: Input data frame with column of sentences.
    :return: Torch matrix of embeddings, size 1024.
    """
    model = skipthoughts.load_model()
    encoder = skipthoughts.Encoder(model)
    #sentences = _df['sentText'].tolist()
    _sent_embs = encoder.encode(sentences, verbose=True)
    _sent_tensors = [torch.from_numpy(j) for j in _sent_embs]
    return torch.stack(_sent_tensors)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="build_glove_space",
                                     description="Builds sentence embeddings from the GloVe model")
    parser.add_argument('-s', '--set', required=True, type=str,
                        help='The dataset name', dest='data')
    args = parser.parse_args()
    wd = os.path.normpath(os.getcwd() + os.sep + os.pardir + os.sep + os.pardir + os.sep + os.pardir + os.sep + os.pardir)
    if args.data == 'riedel':
        sentence_path = os.path.join(wd, "data", "RESIDE", "riedel_data", "sentence_mapper.json")
        with open(sentence_path, 'rb') as f:
            sentences = json.load(f)
    elif args.data == 'gids':
        sentence_path = os.path.join(wd, "data", "GIDS", "gids_data", "sentence_mapper.json")
        with open(sentence_path, 'rb') as f:
            sentences = json.load(f)
    # wd = os.path.normpath(os.getcwd())
    train_sent_space = transform_sentences(sentences)
    torch.save(train_sent_space, '{d}_skipthought_space.pt'.format(d=args.data))
    print(train_sent_space.shape)

