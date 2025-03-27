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


MODE_COV = 'cov'
MODE_VAR = 'var'


def get_gaussian_entropy(cov, base=2):
    # entropy = multivariate_normal.entropy(cov=cov)
    _, logdet = np.linalg.slogdet(2 * np.pi * np.e * cov)
    return 0.5 * logdet / math.log(base)


def get_gaussian_entropy_from_variance(variance, base=2):
    cov = np.diagflat(variance)
    return get_gaussian_entropy(cov, base=base)


def get_row_info(word, word_info, lang, mode='cov'):
    n_samples = word_info['n_samples']
    if lang in constants.LANG_CODE:
        n_senses = len(wn.synsets(word, lang=constants.LANG_CODE[lang]))
    else:
        n_senses = -1

    if mode == MODE_COV:
        covariance = word_info['covariance']
        entropy = get_gaussian_entropy(covariance)
    elif mode == MODE_VAR:
        variance = word_info['variance']
        entropy = get_gaussian_entropy_from_variance(variance)

    return {
        'word': word,
        'n_samples': n_samples,
        'wordnet_polysemy': n_senses,
        'bert_polysemy': entropy,
    }


def get_var_polysemy(lang, filename, save_file):
    variances = utils.read_pickle(filename)
    df = pd.DataFrame(columns=['word', 'n_samples', 'wordnet_polysemy', 'bert_polysemy'])

    for word, word_info in tqdm(variances.items(), desc='Getting variance polysemy'):
        row_info = get_row_info(word, word_info, lang, mode=MODE_VAR)

        df = df.append(row_info, ignore_index=True)

    if lang in constants.LANG_CODE:
        df_temp = df[df.wordnet_polysemy > 0]
        print('BERT vs. Wordnet spearman correlation:', stats.spearmanr(df_temp.wordnet_polysemy, df_temp.bert_polysemy))
    df.to_csv(save_file, sep='\t')


def _get_cov_polysemy(lang, covs_path, save_file, mode=MODE_COV):
    df = pd.DataFrame(columns=['word', 'n_samples', 'wordnet_polysemy', 'bert_polysemy'])

    filenames = utils.get_filenames(covs_path)
    for i, filename in tqdm(list(enumerate(filenames)), desc='Getting covariance polysemy'):
        covs = utils.read_pickle(filename)
        for word, word_info in covs.items():
            row_info = get_row_info(word, word_info, lang, mode=mode)

            if row_info is None:
                continue

            df = df.append(row_info, ignore_index=True)

    if lang in constants.LANG_CODE:
        df_temp = df[df.wordnet_polysemy > 0]
        print('BERT vs. Wordnet spearman correlation:', stats.spearmanr(df_temp.wordnet_polysemy, df_temp.bert_polysemy))
    df.to_csv(save_file, sep='\t')


def get_cov_polysemy(lang, covs_path, save_file):
    _get_cov_polysemy(
        lang, covs_path, save_file, mode=MODE_COV)


def main():
    args = parser.parse_args()
    logging.info(args)

    get_var_polysemy(args.language, args.embeddings_variance_file, args.polysemy_variance_file)
    get_cov_polysemy(args.language, args.embeddings_covariance_path, args.polysemy_covariance_file)


if __name__ == '__main__':
    nltk.download('wordnet')
    nltk.download('omw')
    main()
