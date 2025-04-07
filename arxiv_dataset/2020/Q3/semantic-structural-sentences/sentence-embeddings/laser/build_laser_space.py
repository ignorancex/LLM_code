import argparse
import json
import os
import torch

from laserembeddings import Laser


def transform_sentences(_sent_map):
    """
    Builds sentence embeddings using the LASER model.

    :param _df: Input data frame with column of sentences.
    :return: Torch matrix of embeddings, size 1024.
    """
    laser = Laser()
    sentences = list(_sent_map.keys())
    _sent_embs = laser.embed_sentences(sentences, lang='en')
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
    sent_space = transform_sentences(sentences)
    torch.save(sent_space, '{d}_laser_space.pt'.format(d=args.data))
    print(sent_space.shape)
