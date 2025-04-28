import logging
logging.basicConfig(level=20)
import jax
jax.config.update("jax_enable_x64", True)

import numpy as np
import jax
import argparse
import sys
import os
sys.path.insert(0, "../src")


from svd_2sls import svd_2sls, combine_svd_estimation
from utils import generate_scenario, save_scenario
from optimization_submodular import LinearExperiment
from visualization import plot_three_sets


def run_estimation(seed, n_runs, n, p, name_id, fig_save_path):
    """
    Simple run to estimate finite sample properties; this function is for illustration purpose of the paper

    Parameters
    ----------
    seed: int
    n_runs: int
    n: int
    p: int
    name_id: str
    fig_save_path:str

    Returns
    -------

    """

    key_initial = jax.random.PRNGKey(seed)
    key_scenario, key_finite_sample = jax.random.split(key_initial, 2)

    # ----------------------------------------------------------------------------------------------------------------
    # 1. Generate scenario
    # ----------------------------------------------------------------------------------------------------------------
    d = 3
    key, key_sub = jax.random.split(key_scenario)
    logging.info(
        ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Scenario Generation <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    alpha, beta, e, m = generate_scenario(key_sub, d, p, d)
    save_scenario(key_sub, alpha, beta, e, m, fig_save_path, name_id=name_id)

    # ----------------------------------------------------------------------------------------------------------------
    # 2. Initialize models
    # ----------------------------------------------------------------------------------------------------------------
    # Initialize experiment setting
    iter = 0
    lam1 = np.array([True, False, False])
    lam2 = np.array([False, True, False])
    lam3 = np.array([False, False, True])
    key_chain = jax.random.split(key, n_runs)

    res_beta1, res_beta2, res_beta3 = [], [], []
    res_beta12, res_beta1_2 = [], []
    res_beta123, res_beta12_3, res_beta1_2_3 = [], [], []

    for key in key_chain:
        print(
            "****************************** " + str(iter) + "/" + str(
                n_runs) + " ************************************")
        # ----------------------------------------------------------------------------------------------------------------
        # 3. Estimation
        # ----------------------------------------------------------------------------------------------------------------
        ## Initialize experiment setting
        key_optimization, key_experiment = jax.random.split(key)
        linear_experiment = LinearExperiment(key_experiment, n, alpha, beta, e, m)
        beta_0 = np.hstack([0.0, beta]).squeeze()

        # generate model
        x1, y1, z1 = linear_experiment.run(lam1)
        x2, y2, z2 = linear_experiment.run(lam2)
        x3, y3, z3 = linear_experiment.run(lam3)
        x12, y12, z12 = linear_experiment.run(lam1 + lam2)
        x123, y123, z123 = linear_experiment.run(lam1 + lam2 + lam3)


        beta1, P1, _ = svd_2sls(x1, y1, z1, max_iv=lam1.sum())
        beta2, P2, _ = svd_2sls(x2, y2, z2, max_iv=lam2.sum())
        beta3, P3, _ = svd_2sls(x3, y3, z3, max_iv=lam3.sum())

        beta12, P12, _ = svd_2sls(x12, y12, z12, max_iv=(lam1 + lam2).sum())
        beta123, P123, _ = svd_2sls(x123, y123, z123, max_iv=(lam1 + lam2 + lam3).sum())

        beta1_2 = combine_svd_estimation([beta1, beta2], [P1, P2])
        beta1_2_3 = combine_svd_estimation([beta1, beta2, beta3], [P1, P2, P3])
        beta12_3 = combine_svd_estimation([beta12, beta3], [P12, P3])

        res_beta1.append(beta1), res_beta2.append(beta2), res_beta3.append(beta3)
        res_beta12.append(beta12), res_beta1_2.append(beta1_2)
        res_beta12_3.append(beta12_3), res_beta123.append(beta123), res_beta1_2_3.append(beta1_2_3)

        iter += 1

    res_beta12_3, res_beta123, res_beta1_2_3 = np.array(res_beta12_3), np.array(res_beta123), np.array(res_beta1_2_3)

    # ----------------------------------------------------------------------------------------------------------------
    # 4. Save Path
    # ----------------------------------------------------------------------------------------------------------------
    save_dict = {
        "res_beta123": np.array(res_beta123),
        "res_beta12_3": np.array(res_beta12_3),
        "res_beta1_2_3": np.array(res_beta1_2_3),
        "beta": np.array(beta),
    }
    np.save(os.path.join(fig_save_path, "results.npy"), save_dict)
    # ----------------------------------------------------------------------------------------------------------------
    # 3. Visualization
    # ----------------------------------------------------------------------------------------------------------------
    # CONSTANTS VISUALIZATION
    plot_three_sets(res_beta123, res_beta12_3, res_beta1_2_3, beta_0, name_id, fig_save_path)




if __name__ == '__main__':

    # COMMAND LINE ARGUMENTS
    parser = argparse.ArgumentParser()

    # set up command line arguments
    parser.add_argument("--fig_path", type=str,
                        default="/Users/elisabeth.ailer/Projects/P5_InsufficientIV/Output/FiniteSample_Results")
    parser.add_argument("--n_runs", help="Number of runs for finite sample properties.", type=int, default=1000)

    # Arguments for Algorithm
    parser.add_argument("--p", help="Number of treatment variables", type=int, default=3)
    parser.add_argument("--n", help="Number of samples per experiment", type=int, default=2000)
    parser.add_argument("--seed", help="Seed.", type=int, default=253)

    args = parser.parse_args()
    fig_path = args.fig_path
    p = args.p
    n = args.n
    n_runs = args.n_runs
    seed = args.seed

    # ----------------------------------------------------------------------------------------------------------------
    # 0. Initialization
    # ----------------------------------------------------------------------------------------------------------------
    ## Create directory with name_id
    name_id = "p" + str(p) + "_n_runs" +str(n_runs)
    fig_save_path = os.path.join(fig_path, str(name_id))

    try:
        os.makedirs(fig_save_path)
    except FileExistsError:
        pass

    # ----------------------------------------------------------------------------------------------------------------
    # 1. Estimation routine
    # ----------------------------------------------------------------------------------------------------------------
    run_estimation(seed, n_runs, n, p, name_id, fig_save_path)






