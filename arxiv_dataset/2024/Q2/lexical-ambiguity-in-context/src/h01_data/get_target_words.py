# -*- coding: utf-8 -*-
import sys
from os import path
import logging
import homoglyphs as hg

sys.path.append('./src/')
from utils import argparser as parser
from utils import utils


def get_counts(file):
    counter = {}

    for line in file:
        tokens = line.strip().split(' ')
        if len(tokens) > 100 or len(tokens) <= 2:
            continue

        for token in tokens:
            counter[token] = counter.get(token, 0) + 1

    return counter


def filter_words(language, counts):
    alphabet = utils.get_alphabet(language)
    return {x: count for x, count in counts.items() if utils.is_word(x, alphabet) and len(x) > 1}


def process_file(src_file, language, min_samples):
    with open(src_file, 'r') as f:
        counts = get_counts(f)

    counts = filter_words(language, counts)

    tgt_words = {x for x, count in counts.items() if count >= min_samples}
    tgt_counts = sum([count for x, count in counts.items() if x in tgt_words])
    print(src_file, language)
    print('\tMin samples: %d\t # tokens: %d\t # types: %d' % (min_samples, sum(counts.values()), len(counts)))
    print('\t# tgt types: %d\t # tokens: %d' % (len(tgt_words), tgt_counts))

    return tgt_words


def process(args):
    tgt_words = process_file(args.wikipedia_tokenized_file, args.language, args.min_samples)
    tgt_words &= process_file(args.wikipedia_train_file, args.language, args.min_freq_vocab)

    print('# final types: %d' % (len(tgt_words)))
    utils.write_pickle(args.wikipedia_words_file, tgt_words)


def check_args(args):
    # Check the input files exist
    if not path.isfile(args.wikipedia_tokenized_file):
        logging.error("Tokenized wikipedia file not found: %s", args.wikipedia_tokenized_file)
        sys.exit()

    # Check the input files exist
    if not path.isfile(args.wikipedia_train_file):
        logging.error("Tokenized wikipedia file not found: %s", args.wikipedia_train_file)
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

    process(args)


if __name__ == '__main__':
    main()
