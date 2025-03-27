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


def get_covariances(embs_full, min_samples):
    covs, var = {}, {}
    for word, embs in embs_full.items():
        if embs.shape[0] > min_samples and len(word) > 1:
            covs[word] = {
                'covariance': np.cov(embs, rowvar=False),
                'n_samples': embs.shape[0]
            }
            var[word] = {
                'variance': np.var(embs, axis=0, ddof=1),
                'n_samples': embs.shape[0]
            }
    return covs, var


def save_covs(covs, file_path, file_key):
    tqdm.write('\tSaving file')
    filename = path.join(file_path, 'covs_%s.pickle.bz2' % (file_key))
    utils.write_pickle(filename, covs)


def get_filekeys(filename, data_type):
    base_name = '%s_%s.pickle.bz2' % (data_type, r'(.+?)')
    filename = filename.split('/')[-1]
    m = re.match(base_name, filename)
    return m.group(1)


def main():
    args = parser.parse_args()
    logging.info(args)

    var = {}
    filenames = utils.get_filenames(args.embeddings_merged_path)
    for i, filename in tqdm(list(enumerate(filenames))):
        tqdm.write('%d/%d Reading file: %s. Length of vars: %d' %
                   (i + 1, len(filenames), filename, len(var)))
        embs = utils.read_pickle(filename)
        covs, var_temp = get_covariances(embs, args.min_samples)
        var = dict(var, **var_temp)

        if len(covs) > 0:
            file_key = get_filekeys(filename, 'embs')
            save_covs(covs, args.embeddings_covariance_path, file_key)

    utils.write_pickle(args.embeddings_variance_file, var)


if __name__ == '__main__':
    main()
