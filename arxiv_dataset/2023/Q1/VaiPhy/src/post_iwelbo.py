import os
import sys

sys.path.append(os.getcwd())

import logging
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from main import parse_args
from utils import save, load, compute_loglikelihood
from logging_utils import set_logger, close_logger
from phylo_tree import sample_t_jc


if __name__ == "__main__":
    print("Hello, world!")

    # Parse arguments
    args = parse_args()
    # Add additional arguments required for post analysis preparation

    args.log_filename += "post_iwelbo.log"

    # Set console_level=logging.INFO or console_level=logging.DEBUG
    set_logger(filename=args.log_filename, console_level=logging.INFO)

    if args.model == "vaiphy":  # TODO add other models when needed
        import vaiphy.vaiphy as model

    # Load model
    filename = args.file_prefix + '_object.pkl'
    model = load(filename)
    logging.info("Model is loaded.")
    model.report_edmonds_tree()

    lb_list = []
    for i in range(10):
        np.random.seed(i)
        print("Run: ", i)
        mean_lb, *_ = model.get_iwelbo(log_W=model.log_W_list[-1], t_opts=model.t_opts_list[-1], S=model.S_tensor,
                                       n_iwelbo_samples=model.n_iwelbo_samples, n_iwelbo_runs=model.n_iwelbo_runs)
        print("IWELBO: ", mean_lb)
        lb_list.append(mean_lb)

    print("Mean: ", np.mean(lb_list))
    print("Std: ", np.std(lb_list))

    close_logger()
