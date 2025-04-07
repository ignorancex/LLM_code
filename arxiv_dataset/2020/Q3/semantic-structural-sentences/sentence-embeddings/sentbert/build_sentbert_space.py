import argparse
import json
import os
import sys
import torch

from sentence_transformers import SentenceTransformer


def transform_sentences(_sent_map):
    """
    In the sentence space, we rely entirely on the machinery of the language model, thus there is no way to
    tune the dimensionality of the embeddings. If NOT using sentence transformer and using a raw language model,
    the options for compressing the output to one vector are as follows:
        1. Use the vector associated with the [CLS] token.
        2. Max-pool across each dimension of all output vectors in the tensor.
        3. Take the average across all dimensions for all output vectors in the tensor.
    """
    model = SentenceTransformer('bert-base-nli-mean-tokens')
    sentences = list(_sent_map.keys())
    _sent_embs = model.encode(sentences, show_progress_bar=True)
    _sent_tensors = [torch.from_numpy(j) for j in _sent_embs]
    return torch.stack(_sent_tensors)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="build_sentbert_space",
                                     description="Builds sentence embeddings from the sentbert model")
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
    torch.save(sent_space, '{d}_sentbert_space.pt'.format(d=args.data))
    print(sent_space.shape)
