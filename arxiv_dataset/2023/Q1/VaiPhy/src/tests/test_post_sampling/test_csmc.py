import os
import sys
import time

import pytest
import numpy as np

import utils
from data_utils import read_nex_file
from post_sampling import csmc
from post_sampling.csmc import Particle


@pytest.fixture(scope='module')
def default_particle():
    data = [['A' 'T'], ['A' 'G']]
    n_leaves = len(data)
    pos = len(data[0])
    data = np.array(data).reshape(n_leaves, pos)
    return Particle(idx=0, data=data, branch_sampling_scheme='naive')


def test_jc_likelihood_normal_more_likely_than_huge_edge_length(default_particle):
    b_0 = 0.1
    b_1 = 1000
    beta_draw = 0.5
    expected_counts = [[10, 1, 1, 1], [1, 10, 1, 1], [1, 1, 10, 1], [1, 1, 1, 10]]
    phi = [[expected_counts, expected_counts], [expected_counts, expected_counts]]
    phi = np.array(phi)
    part_0 = default_particle
    log_L_0 = part_0.jc_likelihood(b=b_0, beta_draw=beta_draw, phi=phi)
    log_L_1 = part_0.jc_likelihood(b=b_1, beta_draw=beta_draw, phi=phi)
    assert log_L_0 > log_L_1


def test_jc_likelihood_more_mutations_more_likely_than_few_over_long_edge(default_particle):
    beta_draw = 0.3
    b = - 3 / 4 * np.log(4 / 3 * (beta_draw - 1 / 4))
    expected_counts = [[100, 1, 1, 1], [1, 100, 1, 1], [1, 1, 100, 1], [1, 1, 1, 100]]
    phi = [[expected_counts, expected_counts], [expected_counts, expected_counts]]
    phi = np.array(phi)
    part_0 = default_particle
    log_L_0 = part_0.jc_likelihood(b=b, beta_draw=beta_draw, phi=phi)
    expected_counts = [[1, 100, 100, 100], [100, 1, 100, 100], [100, 100, 1, 100], [100, 100, 100, 1]]
    phi = [[expected_counts, expected_counts], [expected_counts, expected_counts]]
    phi = np.array(phi)
    log_L_1 = part_0.jc_likelihood(b=b, beta_draw=beta_draw, phi=phi)
    assert log_L_0 < log_L_1


def test_csmc_profiling():
    print("Hello, world!")
    data_seed = 1  # 1, 2, 3
    num_taxa = 10  # 10, 20, 40
    n_sites = 300  # 300, 1000
    lambda_ = 4  # 0 if no distortion. 4 or 8 if distorted.

    dataset_name = "data_seed_" + str(data_seed) + "_taxa_" + str(num_taxa) + "_pos_" + str(n_sites)
    results_dir = "../../../results/" + dataset_name \
                  + "/model_vaiphy_init_nj_phyml_samp_slantis_branch_ml_ng_0.8" \
                  + "/S_20_seed_13"

    expected_count_fname = results_dir + "/post_analysis_prep" + "/exp_counts.npy"
    cooccurrence_fname = results_dir + "/post_analysis_prep" + "/node_cooccurrence_mat.npy"
    cooccurrence_fname_dist = results_dir + "/post_analysis_prep" \
                              + "/node_cooccurrence_mat_distorted_lambda_" + str(lambda_) + "_combined.npy"

    # Load Data
    fname = "../../../data/" + dataset_name + "/" + dataset_name + ".nex"
    try:
        data = read_nex_file(fname)
        print("Data shape: ", data.shape)
    except:
        print("Problem occurred during loading data.", fname)
        sys.exit()

    try:
        phi = np.load(expected_count_fname)
        print("Phi shape: ", phi.shape)
    except:
        print("Problem occurred during loading phi.", expected_count_fname)
        sys.exit()

    if lambda_ == 0:
        cooccurrence_fname = results_dir + "/post_analysis_prep/node_cooccurrence_mat.npy"
    else:
        cooccurrence_fname = results_dir + "/post_analysis_prep/node_cooccurrence_mat_distorted_lambda_" \
                             + str(lambda_) + "_combined.npy"
    try:
        c = np.load(cooccurrence_fname)
    except:
        print("Problem occurred during loading co-occurrences.", cooccurrence_fname, lambda_)
        sys.exit()

    print("Distortion rate: ", lambda_)
    print("Co-occurrences shape: ", c.shape)

    bitmask_dict = None
    if lambda_ == 0:
        bitmask_fname = results_dir + "/post_analysis_prep/bitmask_dict.pkl"
    else:
        bitmask_fname = results_dir + "/post_analysis_prep/bitmask_dict_distorted_lambda_" \
                        + str(lambda_) + "_combined.pkl"
    try:
        bitmask_dict = utils.load(bitmask_fname)
    except:
        bitmask_dict = None
        print("Problem occurred during loading bitmask dictionary.", bitmask_fname, lambda_)
        # sys.exit()
    print("Bitmask_dict: ", bitmask_dict)

    results_dir += "/csmc_results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir, exist_ok=True)
    print("Results will be saved to: %s" % results_dir)

    K = 16
    csmc_n_repetitions = 5
    branch_sampling_scheme = 'naive_w_labels'  # 'naive', 'naive_w_labels', 'same_parent', 'prior'
    tree_sampling_scheme = 'uniform'  # 'uniform', 'naive_bootstrap'

    # Run csmc
    start_time_total = time.time()
    print("Running CSMC for data_seed: ", data_seed, " num_taxa", num_taxa, " n_sites: ", n_sites,
          " fname: ", expected_count_fname, " K: ", K)

    marg_logl_estimates = []
    for rep in range(csmc_n_repetitions):
        start_time_rep = time.time()
        out_fname = results_dir + "/branch_" + branch_sampling_scheme + "_tree_" + tree_sampling_scheme + "_K_" + str(K)
        if lambda_ != 0:
            out_fname += "_lambda_" + str(lambda_)

        est, particle_lls = csmc.run(data=data, phi=phi, K=K, c=c, bitmask_dict=bitmask_dict,
                                branch_sampling_scheme=branch_sampling_scheme,
                                tree_sampling_scheme=tree_sampling_scheme, seed_val=rep,
                                is_plot=True, output_directory=out_fname)
        marg_logl_estimates.append(est)
        print("\tRepetition: \t", rep, "\t estimate: \t", est, "\t time taken: \t", time.time() - start_time_rep)
        print("\tLLs. Mean: ", np.mean(particle_lls), " Std: ", np.std(particle_lls), " Max: ", np.max(particle_lls))

    print("\nMean marginal loglikelihood estimate (Z_csmc): ", np.mean(marg_logl_estimates))
    print("Std marginal loglikelihood estimate (Z_csmc): ", np.std(marg_logl_estimates))
    print("Total time taken by CSMCS (sec): ", time.time() - start_time_total)