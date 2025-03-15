# -*- coding: utf-8 -*-
import sys
from os import path
import logging
import torch

sys.path.append('./src/')
# from h01_bert_embeddings.bert import BertProcessor
from h02_bert_embeddings.bert_runner import BertEmbeddingsGetter
from utils import argparser as parser
from utils import utils


def process(src_file, tgt_path, tgt_words, bert_option, batch_size, dump_size):
    bert_runner = BertEmbeddingsGetter(bert_option, batch_size, dump_size, tgt_words)
    bert_runner.get_embeddings(src_file, tgt_path)


def check_args(args):
    # Check the input files exist
    if not path.isfile(args.wikipedia_tokenized_file):
        logging.error("Tokenized wikipedia file not found: %s", args.wikipedia_tokenized_file)
        sys.exit()

    if not path.isfile(args.wikipedia_words_file):
        logging.error("Wikipedia words file not found: %s", args.wikipedia_words_file)
        sys.exit()

    # Check the output folder exist
    if not path.isdir(args.embeddings_raw_path):
        logging.error("Output filename directory does not exist: %s", args.embeddings_raw_path)
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
    process(args.wikipedia_tokenized_file, args.embeddings_raw_path, tgt_words,
            args.bert, args.batch_size, args.dump_size)


if __name__ == '__main__':
    main()
