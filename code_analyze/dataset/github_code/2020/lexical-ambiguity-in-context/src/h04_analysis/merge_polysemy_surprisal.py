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


# MODE_COV = 'cov'
# MODE_VAR = 'var'


# def get_gaussian_entropy(cov, base=2):
#     # entropy = multivariate_normal.entropy(cov=cov)
#     _, logdet = np.linalg.slogdet(2 * np.pi * np.e * cov)
#     return 0.5 * logdet / math.log(base)


# def get_gaussian_entropy_from_variance(variance, base=2):
#     cov = np.diagflat(variance)
#     return get_gaussian_entropy(cov, base=base)


# def get_row_info(word, surprisals, unigram, vocab, word_info, lang, mode='cov'):
#     n_samples = word_info['n_samples']
#     if lang in constants.LANG_CODE:
#         n_senses = len(wn.synsets(word, lang=constants.LANG_CODE[lang]))
#     else:
#         n_senses = -1

#     if word in surprisals:
#         surprisal = sum(surprisals[word]) / len(surprisals[word])
#         s_samples = len(surprisals[word])
#         unigram = unigram[word]
#         in_vocab = (word in vocab)
#     else:
#         surprisal = None
#         s_samples = 0
#         unigram = None
#         in_vocab = None

#     if mode == MODE_COV:
#         covariance = word_info['covariance']
#         entropy = get_gaussian_entropy(covariance)
#     elif mode == MODE_VAR:
#         variance = word_info['variance']
#         entropy = get_gaussian_entropy_from_variance(variance)

#     return {
#         'word': word,
#         'n_samples': n_samples,
#         'n_senses': n_senses,
#         'surprisal': surprisal,
#         's_samples': s_samples,
#         'entropy': entropy,
#         'unigram': unigram,
#         'in_vocab': in_vocab,
#     }


def merge_polysemy_surprisal(polysemy_file, surprisal_file, save_file):
    df_polysemy = pd.read_csv(polysemy_file, sep='\t', index_col=0)
    df_surprisal = pd.read_csv(surprisal_file, sep='\t', index_col=0)

    assert df_surprisal.word.unique().shape[0] == df_surprisal.shape[0]
    assert df_polysemy.word.unique().shape[0] == df_polysemy.shape[0]

    df = pd.concat([df_polysemy.set_index('word'), df_surprisal.set_index('word')], axis=1)
    df.reset_index(inplace=True)
    df.rename(columns = {"index": "word"},
              inplace = True)
    # import ipdb; ipdb.set_trace()
    df.to_csv(save_file, sep='\t')


def get_cov_merge(lang, surprisals, unigram, vocab, covs_path, save_file):
    merge_polysemy_surprisal(polysemy_file, surprisal_file, save_file)
    # _get_cov_correlations(
    #     lang, surprisals, unigram, vocab, covs_path, save_file, mode=MODE_COV)


def get_unigram_surprisals(fname):
    return load_surprisals(fname)


def get_vocab(bert_fname):
    return BertPerWordForMaskedLM.load_vocab(bert_fname)


def main():
    args = parser.parse_args()
    logging.info(args)

    merge_polysemy_surprisal(args.polysemy_variance_file, args.surprisal_file, args.correlation_variance_file)
    merge_polysemy_surprisal(args.polysemy_covariance_file, args.surprisal_file, args.correlation_covariance_file)


if __name__ == '__main__':
    main()
