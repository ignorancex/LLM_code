import sys
import math
import logging
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import wordnet as wn
from scipy import stats
from tqdm import tqdm

sys.path.append('./src/')
from h03_bert_surprisal.bert_mlm_per_word import BertPerWordForMaskedLM
from h03_bert_surprisal.get_unigram_probs import load_surprisals
from utils import argparser as parser
from utils import constants
from utils import utils


def get_row_info(word, surprisals, unigram, vocab):
    surprisal = sum(surprisals) / len(surprisals)
    s_samples = len(surprisals)
    unigram = unigram[word]
    in_vocab = (word in vocab)

    return {
        'word': word,
        'surprisal': surprisal,
        's_samples': s_samples,
        'unigram': unigram,
        'in_vocab': in_vocab,
    }


def get_surprisal_df(lang, surprisals, unigram, vocab, save_file):
    df = pd.DataFrame(columns=['word', 's_samples', 'surprisal', 'unigram', 'in_vocab'])

    for word, surprisal in tqdm(surprisals.items()):
        row_info = get_row_info(word, surprisal, unigram, vocab)
        df = df.append(row_info, ignore_index=True)

    # print('Spearman:', stats.spearmanr(df.surprisal, df.entropy))
    # print('Pearson:', stats.pearsonr(df.surprisal, df.entropy))
    df.to_csv(save_file, sep='\t')


# def _get_cov_correlations(lang, surprisals, unigram, vocab, covs_path, save_file, mode=MODE_COV):
#     df = pd.DataFrame(columns=['word', 'n_samples', 'n_senses', 'entropy', 'surprisal', 's_samples'])

#     filenames = utils.get_filenames(covs_path)
#     for i, filename in tqdm(list(enumerate(filenames))):
#         tqdm.write(
#             '%d/%d Reading file: %s. Number of words: %d.' %
#             (i + 1, len(filenames), filename, df.shape[0]))
#         covs = utils.read_pickle(filename)
#         for word, word_info in covs.items():
#             row_info = get_row_info(word, surprisals, unigram, vocab, word_info, lang, mode=mode)

#             if row_info is None:
#                 # tqdm.write(
#                 #     '\tSkipping word %s which has less than 768 samples in wikipedia' % (word))
#                 continue
#             if row_info['surprisal'] is None:
#                 # tqdm.write('\tSkipping word %s not in surprisals dict' % (word))
#                 continue

#             df = df.append(row_info, ignore_index=True)

#     print('Spearman:', stats.spearmanr(df.surprisal, df.entropy))
#     # print('Pearson:', stats.pearsonr(df.surprisal, df.entropy))
#     df.to_csv(save_file, sep='\t')


# def get_cov_correlations(lang, surprisals, unigram, vocab, covs_path, save_file):
#     _get_cov_correlations(
#         lang, surprisals, unigram, vocab, covs_path, save_file, mode=MODE_COV)


def get_unigram_surprisals(fname):
    return load_surprisals(fname)


def get_vocab(bert_fname):
    return BertPerWordForMaskedLM.load_vocab(bert_fname)


def main():
    args = parser.parse_args()
    logging.info(args)

    surprisals = utils.read_pickle(args.surprisal_bert_file)
    unigram = get_unigram_surprisals(args.unigram_probs_file)
    vocab = get_vocab(args.trained_bert_file)

    get_surprisal_df(args.language, surprisals, unigram, vocab, args.surprisal_file)


if __name__ == '__main__':
    main()
