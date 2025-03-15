# -*- coding: utf-8 -*-
# Process Wikipedia to embeddings, tailored to wikipedia-extractor.py dump output

import sys
from os import path
import argparse
import logging
import multiprocessing as mp
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel, BertForMaskedLM
from torch.functional import F

sys.path.append('./src/')
from utils import argparser as parser
from utils import utils


def merge_dict_lists(dict1, dict2):
    new_dict = {}
    keys = set(dict1.keys()) | set(dict2.keys())
    for key in keys:
        new_vals = np.asarray(dict2.get(key, [])).flatten()
        new_dict[key] = dict1.get(key, []) + new_vals.tolist()

    return new_dict


def process(source_path, output_filename):
    logprobs = {}
    for file in tqdm(utils.get_filenames(source_path), desc='Merging surprisal results'):
        logprobs_single = utils.read_pickle(file)
        logprobs = merge_dict_lists(logprobs, logprobs_single)

    utils.write_pickle(output_filename, logprobs)


def check_args(args):
    # Check the input files exist
    if not path.isdir(args.surprisal_bert_path):
        logging.error("Tokenized wikipedia directory not found: %s", args.surprisal_bert_path)
        sys.exit()

    # Check the output folder exist
    output_path = path.dirname(path.abspath(args.surprisal_bert_file))
    if not path.isdir(output_path):
        logging.error("Output filename directory does not exist: %s", args.surprisal_bert_file)
        sys.exit()


def main():
    args = parser.parse_args()
    logging.info(args)

    check_args(args)

    process(args.surprisal_bert_path, args.surprisal_bert_file)


if __name__ == '__main__':
    main()
