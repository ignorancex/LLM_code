import os
import sys
sys.path.append(os.getcwd())

import logging
import argparse

from data_utils import read_nex_file, convert_data_str_to_onehot
from utils import get_experiment_setup_name
from logging_utils import set_logger, close_logger


def parse_args():
    parser = argparse.ArgumentParser(description='VaiPhy')

    # Data Arguments
    parser.add_argument('--dataset', required=True, help='Dataset name.')
    parser.add_argument('--data_path', default="../data/", help='The data directory.')
    parser.add_argument('--result_path', default="../results/", help='The results directory.')

    # VaiPhy Arguments
    parser.add_argument('--vaiphy_seed', type=int, default=2, help='VaiPhy seed.')
    parser.add_argument('--n_particles', type=int, default=20, help='Number of particles.')
    parser.add_argument('--ng_stepsize', type=float, default=0.1, help='Step size for natural gradient (0, 1].')
    parser.add_argument('--max_iter', type=int, default=100, help='Maximum number of iterations.')
    parser.add_argument('--optimizer', default="ng", help='q(z) update optimizer. adam | ng')

    # VaiPhy evaluation args
    parser.add_argument('--n_iwelbo_samples', type=int, default=10, help='Tempering in the csmc tree sampler. ')
    parser.add_argument('--n_iwelbo_runs', type=int, default=1, help='Tempering in the csmc tree sampler. ')

    # VaiPhy sampling settings
    parser.add_argument('--init_strategy', default="nj_phyml", help='Tree initialization strategy. nj_phyml | random')
    parser.add_argument('--samp_strategy', default="slantis", help='Tree sampling strategy. slantis')
    parser.add_argument('--samp_strategy_eval', default="slantis", help='Tree sampling strategy for eval. slantis | csmc')
    parser.add_argument('--csmc_merg_stategy', default="uniform", help='Tree merger alg in CSMC. (uniform | mst)')
    parser.add_argument('--n_csmc_trees', type=int, default=50, help='Number of trees used inside CSMC to sample one particle.')
    parser.add_argument('--samp_csmc_tempering', type=int, default=1, help='Tempering in the csmc tree sampler. ')
    parser.add_argument('--branch_strategy', default="ml", help='Branch length strategy. ml | jc')
    parser.add_argument('--slantis_explore_rate', type=float, default=0,
                        help='Slantis explore_rate (0 is default Slantis, 1 completely random acceptance probability '
                             'explore)')

    args = parser.parse_args()

    args.model = "vaiphy"  # TODO make it a proper argument if we introduce another model to train, eg viph

    args.data_path += args.dataset + "/"

    # Make folder
    args.result_path += args.dataset
    if not os.path.exists(args.result_path):
        os.makedirs(args.result_path, exist_ok=True)

    setup_details = get_experiment_setup_name(args)
    args.result_path += "/" + setup_details + "/S_" + str(args.n_particles) + "_seed_" + str(args.vaiphy_seed)
    if not os.path.exists(args.result_path):
        os.makedirs(args.result_path, exist_ok=True)

    args.log_filename = args.result_path + "/" + args.model + "_S_" + str(args.n_particles) \
                        + "_seed_" + str(args.vaiphy_seed) + ".log"

    args.file_prefix = args.result_path + '/' + args.model + '_S_' + str(args.n_particles) \
                       + '_seed_' + str(args.vaiphy_seed)

    # TODO Add automatic parameter save function like Vcsmc

    return args


if __name__ == "__main__":
    print("Hello, world!")

    # Parse arguments
    args = parse_args()

    # Load Data
    fname = args.data_path + args.dataset + ".nex"
    try:
        data_raw = read_nex_file(fname)
        data_onehot = convert_data_str_to_onehot(data_raw)
    except:
        sys.exit("Problem occurred during loading data. %s" % fname)
    args.n_taxa, args.n_pos = data_raw.shape
    print("\nLoaded Taxa = ", str(args.n_taxa), "\tPos = ", str(args.n_pos))

    # Set console_level=logging.INFO or console_level=logging.DEBUG
    set_logger(filename=args.log_filename, console_level=logging.INFO)

    logging.info("Parameters. taxa: %d,  n_pos: %d, S: %d, max_iter: %d, vaiphy_seed: %d, "
                 "tree_init: %s, tree_sampling: %s, branch: %s, ng_stepsize: %s"
                 % (args.n_taxa, args.n_pos, args.n_particles, args.max_iter, args.vaiphy_seed, args.init_strategy,
                    args.samp_strategy, args.branch_strategy, args.ng_stepsize))

    if args.model == "vaiphy":  # TODO add other models when needed
        import vaiphy.vaiphy as model

    model = model.MODEL(data_raw=data_raw, data_onehot=data_onehot, args=args)
    logging.info("Model is initialized.")

    if model.true_tree_loglikelihood is not None:
        logging.info("True Tree Log-Likelihood: %s" % str(model.true_tree_loglikelihood))

    model.train(is_print=True)
    logging.info("Training is finished.")

    model.analysis()
    logging.info("Analysis is finished.")

    close_logger()
