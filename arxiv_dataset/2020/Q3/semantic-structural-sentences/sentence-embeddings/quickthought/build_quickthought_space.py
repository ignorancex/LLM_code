import argparse
import json
import numpy as np
import os
import pandas as pd
import torch

from tensorflow.python.framework.errors_impl import InvalidArgumentError
from encoder_manager_upgrade import EncoderManager
from tqdm import tqdm


def transform_sentences(sentences, _mc):
    #sentences = _df['sentText'].tolist()
    manager = EncoderManager()
    manager.load_model(_mc)
    _sent_embs = []
    num_skip = 0
    _sent_embs = manager.encode(sentences, use_norm=False).astype(np.float64)
    #for sent in tqdm(sentences):
    #    try:
    #        encodings = manager.encode([sent]).astype(np.float64)
    #    except (InvalidArgumentError, AssertionError) as e:
    #        num_skip += 1
    #        encodings = np.zeros(2400).astype(np.float64).reshape(1, 2400)
    #    assert encodings.shape == (1, 2400)
    #    _sent_embs.append(np.nan_to_num(encodings))
    #print("Fetching sentence embeddings failed {n} times".format(n=num_skip))
    _sent_tensors = [torch.from_numpy(np.nan_to_num(j)).squeeze() for j in _sent_embs]
    return torch.stack(_sent_tensors)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="build_glove_space",
                                     description="Builds sentence embeddings from the GloVe model")
    parser.add_argument('-s', '--set', required=True, type=str,
                        help='The dataset name', dest='data')
    args = parser.parse_args()
    wd = os.path.normpath(os.getcwd() + os.sep + os.pardir + os.sep + os.pardir + os.sep + os.pardir + os.sep + os.pardir + os.sep + os.pardir)
    if args.data == 'riedel':
        sentence_path = os.path.join(wd, "data", "RESIDE", "riedel_data", "sentence_mapper.json")
        with open(sentence_path, 'rb') as f:
            sentences = json.load(f)
    elif args.data == 'gids':
        sentence_path = os.path.join(wd, "data", "GIDS", "gids_data", "sentence_mapper.json")
        with open(sentence_path, 'rb') as f:
            sentences = json.load(f)
    mc = {"encoder": "gru",
          "encoder_dim": 1200,
          "bidir": True,
          "case_sensitive": True,
          "checkpoint_path": "BS400-W300-S1200-UMBC-bidir/train",
          "vocab_configs": [
              {
                  "mode": "expand",
                  "name": "word_embedding",
                  "cap": False,
                  "dim": 300,
                  "size": 941937,
                  "vocab_file": "BS400-W300-S1200-UMBC-bidir/exp_vocab/word_embedding",
                  "embs_file": "BS400-W300-S1200-UMBC-bidir/exp_vocab/word_embedding"
              }
          ]
          }
    wd = "~/PycharmProjects/paired-geometric-invariance/src"
    train_sent_space = transform_sentences(sentences, mc)
    torch.save(train_sent_space, '../../{d}_quickthought_space.pt'.format(d=args.data))
    print(train_sent_space.shape)
