import os
import sys
sys.path.append(os.getcwd())

import time
import logging
import argparse
import numpy as np
from multiprocessing import Pool, Manager

from post_sampling.csmc import run
from phylo_tree import PhyloTree
from data_utils import read_nex_file
from bifurcate_utils import bifurcate_tree
from tree_utils import newick2nx, nx_to_newick, update_topology, save_consensus_tree
from utils import get_experiment_setup_name, save, load, compute_loglikelihood
from logging_utils import set_logger, close_logger


if __name__ == "__main__":
    print("Hello, world!")

    parser = argparse.ArgumentParser()

    # Data Arguments
    parser.add_argument('--dataset', required=True, help='Dataset path.')

    # VaiPhy Arguments
    parser.add_argument('--vaiphy_seed', type=int, default=2, help='Vaiphy seed.')
    parser.add_argument('--n_particles', type=int, default=20, help='Number of particles.')
    parser.add_argument('--ng_stepsize', type=float, default=0.1, help='Step size for natural gradient (0, 1].')

    parser.add_argument('--init_strategy', default="nj_phyml", help='Tree initialization strategy. nj_phyml | random')
    parser.add_argument('--samp_strategy', default="slantis", help='Tree sampling strategy. slantis')
    parser.add_argument('--branch_strategy', default="ml", help='Branch length strategy. ml | jc')

    parser.add_argument('--data_path', default="../data/", help='The data directory.')

    # Csmc Arguments
    parser.add_argument('--csmc_seed', type=int, default=0, help='Seed offset for Csmc.')
    parser.add_argument('--csmc_n_particles', type=int, default=256, help='Number of particles in Csmc.')
    parser.add_argument('--csmc_n_repetitions', type=int, default=10, help='Number of repetitions in Csmc.')
    parser.add_argument('--csmc_branch_strategy', default="all",
                        help='Csmc branch sampling strategy. all | naive | naive_w_labels | same_parent | prior')
    parser.add_argument('--csmc_tree_strategy', default="all",
                        help='Csmc tree sampling strategy. all | uniform | naive_bootstrap_X | bootstrap_X | naive_iw')
    parser.add_argument('--csmc_distortion_poisson_rate', type=int, default=0,
                        help='Poisson distortion rate in Csmc. 0 | 4 | 8')
    parser.add_argument('--csmc_result_path', default="../results/", help='The results directory for CSMC.')

    args = parser.parse_args()
    args.model = 'vaiphy'  # TODO make it proper argument

    args.csmc_result_path += args.dataset
    if not os.path.exists(args.csmc_result_path):
        os.makedirs(args.csmc_result_path, exist_ok=True)

    setup_details = get_experiment_setup_name(args)
    args.csmc_result_path += "/" + setup_details + "/S_" + str(args.n_particles) + "_seed_" + str(args.vaiphy_seed)
    if not os.path.exists(args.csmc_result_path):
        os.makedirs(args.csmc_result_path, exist_ok=True)
    args.csmc_result_path_2 = args.csmc_result_path + "/csmc_results"
    if not os.path.exists(args.csmc_result_path_2):
        os.makedirs(args.csmc_result_path_2, exist_ok=True)
    #print(args.csmc_result_path_2)

    # Set console_level=logging.INFO or console_level=logging.DEBUG
    log_filename = args.csmc_result_path + "/csmc_vaiphy_S_" + str(args.n_particles) \
                   + "_seed_" + str(args.vaiphy_seed) + "_K_" + str(args.csmc_n_particles) \
                   + "_poisson_" + str(args.csmc_distortion_poisson_rate) + "_rep_" + str(args.csmc_n_repetitions) \
                   + "_" + args.csmc_branch_strategy + "_" + args.csmc_tree_strategy \
                   + "_csmcseed_" + str(args.csmc_seed) + ".log"
    set_logger(filename=log_filename, console_level=logging.INFO)
    #print("log_filename: ", log_filename)

    # Load Data
    filename = args.data_path + args.dataset + "/" + args.dataset + ".nex"
    try:
        data = read_nex_file(filename)
        logging.info("Data shape: %s, %s" % (data.shape[0], data.shape[1]))
    except:
        logging.info("Problem occurred during loading data. %s" % filename)
        sys.exit()

    n_taxa = data.shape[0]
    root = 2 * n_taxa - 3

    # Parse branch sampling strategies
    if args.csmc_branch_strategy is "all":
        branch_sampling_scheme_list = ["naive_w_labels", "same_parent", "prior"]  # 'naive' not involved
    else:
        branch_sampling_scheme_list = [args.csmc_branch_strategy]
    # Parse tree sampling strategies
    if args.csmc_tree_strategy is "all":
        tree_sampling_scheme_list = ["uniform", "naive_bootstrap_0", "bootstrap_0", "naive_iw"]
    else:
        tree_sampling_scheme_list = [args.csmc_tree_strategy]

    # Csmc Part
    # Load expected counts
    expected_count_fname = args.csmc_result_path + "/post_analysis_prep/exp_counts.npy"
    try:
        phi = np.load(expected_count_fname)
        logging.info("Phi shape: %s, %s" % (phi.shape[0], phi.shape[1]))
    except:
        logging.info("Problem occurred during loading phi. %s" % expected_count_fname)
        sys.exit()

    # Load cooccurrence matrix
    if args.csmc_distortion_poisson_rate == 0:
        cooccurrence_fname = args.csmc_result_path + "/post_analysis_prep/node_cooccurrence_mat.npy"
    else:
        cooccurrence_fname = args.csmc_result_path + "/post_analysis_prep/node_cooccurrence_mat_distorted_lambda_" \
                             + str(args.csmc_distortion_poisson_rate) + "_combined.npy"
    try:
        c = np.load(cooccurrence_fname)
    except:
        logging.info("Problem occurred during loading co-occurrences. %s, %s"
                     % (cooccurrence_fname, args.csmc_distortion_poisson_rate))
        sys.exit()

    # Load bitmask dictionary
    bitmask_dict = None
    if args.csmc_distortion_poisson_rate == 0:
        if args.csmc_tree_strategy == "naive_iw":
            bitmask_fname = args.csmc_result_path + "/post_analysis_prep/bitmask_trees_dict.pkl"
        else:
            bitmask_fname = args.csmc_result_path + "/post_analysis_prep/bitmask_dict.pkl"
    else:
        if args.csmc_tree_strategy == "naive_iw":
            bitmask_fname = args.csmc_result_path + "/post_analysis_prep/bitmask_trees_dict_distorted_lambda_" \
                            + str(args.csmc_distortion_poisson_rate) + "_combined.pkl"
        else:
            bitmask_fname = args.csmc_result_path + "/post_analysis_prep/bitmask_dict_distorted_lambda_" \
                            + str(args.csmc_distortion_poisson_rate) + "_combined.pkl"
    try:
        bitmask_dict = load(bitmask_fname)
    except:
        bitmask_dict = None
        logging.info("Problem occurred during loading bitmask dictionary. %s, %s"
                     % (bitmask_fname, args.csmc_distortion_poisson_rate))

    #logging.info("Bitmask_dict: %s" % bitmask_dict)
    logging.info("Distortion rate: %s" % args.csmc_distortion_poisson_rate)
    logging.info("Co-occurrences shape: %s, %s" % (c.shape[0], c.shape[1]))
    logging.info("Number of repetitions: %s" % args.csmc_n_repetitions)
    logging.info("Number of VaiPhy particles: %s" % args.n_particles)
    logging.info("Branch sampling strategy: %s" % branch_sampling_scheme_list)
    logging.info("Tree sampling strategy: %s" % tree_sampling_scheme_list)

    # Run Csmc
    for branch_sampling_scheme in branch_sampling_scheme_list:
        for tree_sampling_scheme in tree_sampling_scheme_list:
            start_time_total = time.time()
            logging.info("Running CSMC for dataset: %s phi_fname: %s " % (args.dataset, expected_count_fname))
            logging.info("K: %s branch_sampling_scheme: %s tree_sampling_scheme: %s"
                         % (args.csmc_n_particles, branch_sampling_scheme, tree_sampling_scheme))

            marg_logl_estimates = []

            out_fname = args.csmc_result_path_2 + "/branch_" + branch_sampling_scheme + "_tree_" \
                        + tree_sampling_scheme + "_K_" + str(args.csmc_n_particles)
            if args.csmc_distortion_poisson_rate != 0:
                out_fname += "_lambda_" + str(args.csmc_distortion_poisson_rate)

            """
            merger_memory = {}
            all_particles = []
            all_particle_lls = []
            for rep in range(args.csmc_n_repetitions):
                print("rep: ", rep, " mem size: ", len(list(merger_memory.keys())))
                start_time_rep = time.time()
                out_fname = args.csmc_result_path_2 + "/branch_" + branch_sampling_scheme \
                            + "_tree_" + tree_sampling_scheme + "_K_" + str(args.csmc_n_particles)
                if args.csmc_distortion_poisson_rate != 0:
                    out_fname += "_lambda_" + str(args.csmc_distortion_poisson_rate)

                est, particle_lls, particles = run(data=data, phi=phi, K=args.csmc_n_particles, c=c, bitmask_dict=bitmask_dict, merger_memory=merger_memory,
                                        branch_sampling_scheme=branch_sampling_scheme,
                                        tree_sampling_scheme=tree_sampling_scheme, seed_val=rep,
                                        is_plot=True, output_directory=out_fname)
                marg_logl_estimates.append(est)
                all_particle_lls += particle_lls
                all_particles += particles
                print("\tRepetition: \t", rep, "\t estimate: \t", est, "\t time taken: \t", time.time() - start_time_rep,
                      flush=True)
                print("\tLLs. Mean: ", np.mean(particle_lls), " Std: ", np.std(particle_lls),
                      " Max: ", np.max(particle_lls), flush=True)
            print("rep: ", rep, " mem size: ", len(list(merger_memory.keys())))
            """
            with Pool(args.csmc_n_repetitions) as p:
                output = p.starmap(run, [
                    (data, phi, args.csmc_n_particles, c, bitmask_dict, 'vaiphy', branch_sampling_scheme,
                     tree_sampling_scheme, rep+args.csmc_seed, False, False, out_fname) for rep in range(args.csmc_n_repetitions)])

                all_particles = []
                all_particle_lls = []
                for est, particle_lls, particles in output:
                    marg_logl_estimates.append(est)
                    all_particle_lls += particle_lls
                    all_particles += particles
                    logging.info("\tEstimate: %s" % est)
                    logging.info("\tLLs. Mean: %s Std: %s Max: %s"
                                 % (np.mean(particle_lls), np.std(particle_lls), np.max(particle_lls)))
            #"""
            logging.info("Mean marginal loglikelihood estimate (Z_csmc): %s" % np.mean(marg_logl_estimates))
            logging.info("Std marginal loglikelihood estimate (Z_csmc): %s" % np.std(marg_logl_estimates))
            logging.info("Particle LLs Mean: %s Std: %s Max: %s"
                         % (np.mean(all_particle_lls), np.std(all_particle_lls), np.max(all_particle_lls)))
            logging.info("Total time taken by CSMCS (sec) for %s, %s : %s"
                         % (branch_sampling_scheme, tree_sampling_scheme, time.time() - start_time_total))

            best_particle_idx = np.argmax(all_particle_lls)
            best_tree = all_particles[best_particle_idx].T
            update_topology(best_tree, root)
            logging.info("Best particle's LL: \t %s" % all_particle_lls[best_particle_idx])
            logging.info("Best particle's Newick: \t %s" % nx_to_newick(best_tree, root))

            # Extract particle trees
            all_particle_trees = []
            all_particle_trees_bifurcated = []
            for particle in all_particles:
                cur_tree = particle.T
                update_topology(cur_tree, root)
                all_particle_trees.append(cur_tree)
                all_particle_trees_bifurcated.append(bifurcate_tree(cur_tree, n_taxa))

            # Save trees
            fname = out_fname + "_all_trees.pkl"
            save(all_particle_trees, fname)
            logging.info("Particle trees are saved to %s" % fname)

            phylo = PhyloTree(data)
            # Majority consensus
            for cutoff in [0.5, 0.1]:
                fname = out_fname + "_majority_consensus_cutoff_" + str(cutoff)
                consensus_newick = save_consensus_tree(fname, root=root,
                                                       tree_list=all_particle_trees_bifurcated, cutoff=cutoff)
                logging.info("Consensus tree Newick (cutoff=%s): \t%s" % (cutoff, consensus_newick))
                consensus_tree = newick2nx(consensus_newick, n_taxa)
                update_topology(consensus_tree, root)
                loglik = compute_loglikelihood(phylo.compute_up_messages(consensus_tree))
                logging.info("Consensus tree's Log-Likelihood (cutoff=%s): \t %s" % (cutoff, loglik))

    close_logger()
