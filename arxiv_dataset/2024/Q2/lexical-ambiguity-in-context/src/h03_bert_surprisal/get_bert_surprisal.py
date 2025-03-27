# -*- coding: utf-8 -*-
import sys
from os import path
import logging
import torch

sys.path.append('./src/')
from h03_bert_surprisal.bert_runner import BertSurprisalGetter
from utils import argparser as parser
from utils import utils


def process(source_file, bert_fname, output_path, tgt_words, bert_option, batch_size, dump_size):
    bert_runner = BertSurprisalGetter(bert_fname, bert_option, batch_size, dump_size, tgt_words)
    bert_runner.get_surprisals(source_file, output_path)


def check_args(args):
    # Check the input files exist
    if not path.isfile(args.wikipedia_tokenized_file):
        logging.error("Tokenized wikipedia directory not found: %s", args.wikipedia_tokenized_file)
        sys.exit()

    # Check the input files exist
    if not path.isfile(args.wikipedia_words_file):
        logging.error("Tokenized wikipedia directory not found: %s", args.wikipedia_words_file)
        sys.exit()

    # Check that bert file exists
    if not path.isfile(args.trained_bert_file):
        logging.error("Trained bert filename does not exist: %s", args.trained_bert_file)
        sys.exit()

    # Check the output folder exist
    if not path.isdir(args.surprisal_bert_path):
        logging.error("Output filename directory does not exist: %s", args.surprisal_bert_path)
        sys.exit()

    # Sanity check the BERT model
    if args.bert not in {'bert-base-uncased', 'bert-large-uncased', 'bert-base-cased',
                         'bert-large-cased', 'bert-base-multilingual-cased', 'bert-base-chinese',
                         'bert-base-german-cased'}:
        logging.error("Invalid BERT model. See https://huggingface.co/transformers/pretrained_models.html")
        sys.exit()


def main():
    args = parser.parse_args()
    logging.info(args)

    check_args(args)

    tgt_words = utils.read_pickle(args.wikipedia_words_file)
    process(args.wikipedia_tokenized_file, args.trained_bert_file, args.surprisal_bert_path, tgt_words, args.bert,
            args.batch_size, args.dump_size)


if __name__ == '__main__':
    main()
