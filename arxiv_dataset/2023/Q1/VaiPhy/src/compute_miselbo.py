import os
import sys
sys.path.append(os.getcwd())

from tree_utils import update_topology
import numpy as np

from scipy.special import logsumexp
from utils import compute_loglikelihood, beta_log_pdf, load
from phylo_tree import sample_t_jc
from post_sampling.csmc import sample_trees as sample_trees_csmc


def get_miselbo(models, n_iwelbo_samples=10, n_iwelbo_runs=100, seed=0):
    # the log-probability of the observations given the trees is computed using DP
    # the probability of sampling the map-estimate from the map-estimate is 1, i.e. q(Lambda) = 1
    # log_qtree: the unnormalized log-importance weights of each tree in the particle system
    # prior over branch lengths is Exp(10), where 10 is the expectation, i.e. rate = 0.1
    # the prior over trees is uniform, the number of possible trees are (2*n_leaves - 3)!!
    np.random.seed(seed)

    all_lbs = []
    M = len(models)
    log_ptree = - np.sum(np.log(np.arange(3, models[0].n_nodes-1, 2)))
    # n_nodes, n_taxa = models[0].n_nodes, models[0].n_taxa
    # n_internals = n_nodes - n_taxa
    # log_ptree = -(n_nodes - 2) * np.log(n_internals) + np.sum(np.log(np.arange(1, n_internals + 1)))
    S_mat = np.zeros((M, *models[0].S_tensor.shape))

    for m, model in enumerate(models):
        S_mat[m] = model.S_tensor

    for l in range(n_iwelbo_runs):
        log_ratio = np.zeros((n_iwelbo_samples, M))
        for m, model in enumerate(models):
            log_q_mixture = np.zeros((n_iwelbo_samples, M))
            log_W = model.log_W_list[-1]
            t_opts = model.t_opts_list[-1]
            # sampled_trees, log_qtree = model.sample_trees(n_iwelbo_samples, log_W, t_opts, return_q=True)
            sampled_trees = []
            log_qtree = []
            for i in range(0, n_iwelbo_samples):
                tau, log_pi, log_Z = sample_trees_csmc(model.data, log_W, t_opts, K=250, phi=model.S_tensor)
                update_topology(tau, model.root)
                sampled_trees.append(tau)
                log_qtree.append(log_pi - log_Z)
            log_qtree = np.array(log_qtree)
            for k, T in enumerate(sampled_trees):
                log_q_mixture[k] += log_qtree[k]
                log_plambda = 0
                for u, v in T.edges:
                    T[u][v]["weight"] = log_W[u, v]
                    T[u][v]["t"], beta_draw = sample_t_jc(S_ij=S_mat[m, u, v], return_p=True)

                    log_df_dt = - 4 / 3 * T[u][v]["t"]
                    log_plambda += np.log(1 / 0.1) - 1 / 0.1 * T[u][v]["t"]
                    for m_prime, phi in enumerate(S_mat):
                        S_ij = phi[u, v]
                        m_same = np.trace(S_ij)
                        m_diff = np.sum(S_ij) - m_same
                        a_param = m_same + 1
                        b_param = m_diff + 1
                        beta_draw_log_likelihood = beta_log_pdf(beta_draw, a_param, b_param)
                        log_q_mixture[k, m_prime] += beta_draw_log_likelihood + log_df_dt

                up_table = model.phylo.compute_up_messages(T.copy())
                log_py_lambdatree = compute_loglikelihood(up_table.copy())
                log_ratio[k, m] = log_py_lambdatree + log_plambda + log_ptree - logsumexp(log_q_mixture[k] - np.log(M), axis=-1)
        iwelbo = np.mean(logsumexp(log_ratio - np.log(n_iwelbo_samples), axis=0))
        all_lbs.append(iwelbo)

    all_lbs = np.array(all_lbs)
    mean_lb = np.mean(all_lbs)
    std_lb = np.std(all_lbs)
    return mean_lb, std_lb


if __name__ == "__main__":
    print("Hello, world!")


    f1 = "../results/DS1/model_vaiphy_init_nj_phyml_samp_slantis_branch_jc_ng_0.1/S_50_seed_13/vaiphy_S_50_seed_13_object.pkl"
    f2 = "../results/DS1/model_vaiphy_init_nj_phyml_samp_slantis_branch_jc_ng_0.1/S_50_seed_0/vaiphy_S_50_seed_0_object.pkl"
    f3 = "../results/DS1/model_vaiphy_init_nj_phyml_samp_slantis_branch_jc_ng_0.09/S_50_seed_1/vaiphy_S_50_seed_1_object.pkl"
    f_list = [f1]  # [f1, f2]
    models = []

    for filename in f_list:
        model = load(filename)
        models.append(model)
        mean_lb, std_lb = get_miselbo([model], n_iwelbo_samples=1, n_iwelbo_runs=1, seed=0)
        print(mean_lb)

    mean_lb, std_lb = get_miselbo(models,  n_iwelbo_samples=5, n_iwelbo_runs=1, seed=0)
    print("MISELBO: ", mean_lb, "pm ", std_lb)
