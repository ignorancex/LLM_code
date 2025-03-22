# -*- coding: utf-8 -*-
import sys
from os import path
import logging
import re
import numpy as np
from tqdm import tqdm

sys.path.append('./src/')
from utils import argparser as parser
from utils import utils


def split_embeddings_per_file(embs_orig, n_chars=2):
    '''
    Process the embeddings in such a way to save it in a new `File system`.
    All embeddings for a word will be in the same file. Listed in alphabetical order.

    n_chars: Nubmer of characters to consider when splitting words in files.
    '''
    embs_filesystem = {}
    for word_orig, embs_word in embs_orig.items():
        word = word_orig.lower()
        if len(word) <= 1:
            continue

        file_key = word[:n_chars]
        if file_key not in embs_filesystem:
            embs_filesystem[file_key] = {}
        embs_filesystem[file_key][word] = np.matrix(embs_word)
    return embs_filesystem


def merge_embeddings(embs1, embs2):
    for word, embs2_word in embs2.items():
        if word not in embs1:
            embs1[word] = embs2_word
        else:
            embs1[word] = np.concatenate([embs1[word], embs2_word])
    return embs1


def merge_embeddings_full(embs_full1, embs_full2):
    for file_key, embs2 in embs_full2.items():
        if file_key not in embs_full1:
            embs_full1[file_key] = embs2
        else:
            embs1 = embs_full1[file_key]
            embs_full1[file_key] = merge_embeddings(embs1, embs2)
    return embs_full1


def merge_embeddings_into_files(file_key, embs_new, save_path):
    filename = path.join(save_path, 'embs_%s.pickle.bz2' % (file_key))
    embs_file = utils.read_pickle_if_exists(filename)
    embs_file = merge_embeddings(embs_file, embs_new)
    utils.write_pickle(filename, embs_file)


def save_embeddings_in_files_per_word(embs, save_path):
    '''
    Process the embeddings in such a way to save it in a new `File system`.
    All embeddings for a word will be in the same file. Listed in alphabetical order.

    n_chars: Nubmer of characters to consider when splitting words in files.
    '''
    tqdm.write('\tSaving files')
    for file_key, embs_single in embs.items():
        merge_embeddings_into_files(file_key, embs_single, save_path)


def get_embeddings_in_files_per_word(filename, n_chars=2):
    '''
    Process the embeddings in such a way to save it in a new `File system`.
    All embeddings for a word will be in the same file. Listed in alphabetical order.

    n_chars: Nubmer of characters to consider when splitting words in files.
    '''
    embs = utils.read_pickle(filename)
    embs = split_embeddings_per_file(embs, n_chars=n_chars)
    return embs


def main():
    args = parser.parse_args()
    logging.info(args)

    filenames = utils.get_filenames(args.embeddings_raw_path)
    embs = {}
    for i, filename in tqdm(list(enumerate(filenames))):
        embs_file = get_embeddings_in_files_per_word(filename, n_chars=args.n_chars_filesystem)
        embs = merge_embeddings_full(embs, embs_file)

        if ((i + 1) % args.dump_size) == 0:
            save_embeddings_in_files_per_word(
                embs, args.embeddings_merged_path)
            embs = {}

    if embs:
        save_embeddings_in_files_per_word(
            embs, args.embeddings_merged_path)


if __name__ == '__main__':
    main()
