# -*- coding: utf-8 -*-
import sys
from os import path
import logging
import torch

sys.path.append('./src/')
from h03_bert_surprisal.bert_trainer import BertMLMTrainer
# from h03_bert_surprisal.bert_runner_gpu import BertSurprisalGetterGPU
from utils import argparser as parser


def process(source_file, output_path, bert_option, batch_size, dump_size, min_freq_vocab):
    bert_runner = BertMLMTrainer(bert_option, batch_size, min_freq_vocab)
    bert_runner.train(source_file, output_path)


def check_args(args):
    # Check the input file exists
    if not path.isfile(args.wikipedia_train_file):
        logging.error("Tokenized train wikipedia file not found: %s", args.wikipedia_train_file)
        sys.exit()

    # Check the output folder exist
    trained_bert_path = path.dirname(path.abspath(args.trained_bert_file))
    if not path.isdir(trained_bert_path):
        logging.error("Output filename directory does not exist: %s", args.trained_bert_file)
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

    process(args.wikipedia_train_file, args.trained_bert_file, args.bert,
            args.batch_size, args.dump_size, args.min_freq_vocab)


if __name__ == '__main__':
    main()
