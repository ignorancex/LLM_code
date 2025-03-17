import argparse
import pickle
import sys

import pandas as pd
import numpy as np

import data_utils
import post_sampling.csmc as csmc
from data_utils import read_nex_file


def parse_args():
    parser = argparse.ArgumentParser(
                        description='Variational Combinatorial Sequential Monte Carlo')
    parser.add_argument('--dataset',
                        help='benchmark dataset to use: primate_data, DS1',
                        default='primate_data')
    parser.add_argument('--method',
                        help='optimization method used',
                        default='vaiphy')
    parser.add_argument('--seed',
                        type=int,
                        help='seed to use',
                        default=0)
    parser.add_argument('--K',
                        type=int,
                        help='number of particles to use',
                        default=8)
    parser.add_argument('--verbose',
                        action='store_true',
                        help='enables verbose output during run time')
    args = parser.parse_args()
    return args

Alphabet_dir_blank = {'A': [1, 0, 0, 0],
                          'C': [0, 1, 0, 0],
                          'G': [0, 0, 1, 0],
                          'T': [0, 0, 0, 1],
                          '-': [1, 1, 1, 1],
                          '?': [1, 1, 1, 1]}


def get_vcsmc_hohna_dataset(dataset):
    datadict_raw = pd.read_pickle("../data/hohna_datasets/" + dataset + ".pickle")
    genome_strings = list(datadict_raw.values())
    datadict = data_utils.form_dataset_from_strings(genome_strings)
    return datadict["genome"], datadict


if __name__ == "__main__":
    args = parse_args()

    np.random.seed(args.seed)
    if "DS" in args.dataset:
        data, _ = get_vcsmc_hohna_dataset(args.dataset)

    path = "../results/vcsmc/hohna_data_1/2048/jc/test_run/"
    K = args.K
    phi = pd.read_pickle(path + "branch_params.p")
    logZ = csmc.run(data, phi, K, method=args.method)
    print(f"logZ: {logZ}")
    # TODO: Comparison
    # stored_res = pd.read_pickle(path + "results.p")