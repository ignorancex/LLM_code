import bz2
import pickle
from os import listdir, path
import pathlib
import numpy as np
import torch
import homoglyphs as hg

from utils import constants


def get_filenames(filepath):
    filenames = [path.join(filepath, f)
                 for f in listdir(filepath)
                 if path.isfile(path.join(filepath, f))]
    return sorted(filenames)


def write_pickle(filename, embeddings):
    with open(filename, "wb") as f:
        pickle.dump(embeddings, f)


def read_pickle(filename):
    with open(filename, "rb") as f:
        embeddings = pickle.load(f)
    return embeddings


def read_pickle_if_exists(filename):
    try:
        return read_pickle(filename)
    except FileNotFoundError:
        return {}


def get_n_lines(fname):
    with open(fname, 'r') as file:
        count = 0
        for _ in file:
            count += 1
    return count


def get_alphabet(language):
    try:
        alphabet = hg.Languages.get_alphabet([language])
    except ValueError:
        if constants.scripts[language] is not None:
            alphabet = hg.Categories.get_alphabet(
                [x.upper() for x in constants.scripts[language]]
            )
        else:
            alphabet = None

    return alphabet


def is_word(token, alphabet):
    return all([x in alphabet for x in token])


def config(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def mkdir(folder):
    pathlib.Path(folder).mkdir(parents=True, exist_ok=True)
