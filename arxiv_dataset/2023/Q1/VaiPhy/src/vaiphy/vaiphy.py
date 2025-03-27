import os
import sys

import utils

sys.path.append(os.getcwd())

import time
import logging
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.special import logsumexp
from utils import save, get_experiment_setup_name, alphabet_size, nuc_vec, compute_loglikelihood
from phylo_tree import PhyloTree, sample_t_jc
from mst_utils import get_k_best_mst
from bifurcate_utils import bifurcate_tree
from tree_utils import newick2nx, nx_to_newick, nx2ete, ete_compare, draw_dendrogram, \
    create_tree, save_tree_image, get_likelihood_from_newick_file, update_topology, save_consensus_tree
from utils import beta_log_pdf
from post_sampling.csmc import sample_trees as sample_trees_csmc


class MODEL:
    def __init__(self, data_raw, data_onehot, args):
        self.model = "vaiphy"
        self.data = data_raw                        # Stores sequence data in raw form, ACGT
        self.data_onehot = data_onehot              # Stores sequence data in one-hot form, [1,0,0,0]  # TODO remove unnecessary conversions from other functions
        self.n_taxa, self.n_pos = self.data.shape   # Stores number of leaves and sequence length
        self.n_nodes = 2 * self.n_taxa - 2          # Stores number of nodes
        self.root = 2 * self.n_taxa - 3             # Stores id of arbitrary root node

        self.seed = args.vaiphy_seed                # Stores the seed value for reproducibility
        self.max_iter = args.max_iter               # Stores the maximum number of iterations
        self.ng_stepsize = args.ng_stepsize         # Stores the stepsize for natural gradient

        self.phylo = PhyloTree(self.data)           # Stores PhyloTree object
        self.phylo.initialize_leaf_links()          # Initializes leaf links

        self.g = np.ones((self.n_nodes, self.n_pos, alphabet_size)) * (1 / alphabet_size)  # Stores current g(Z) matrix, uniform initialization
        self.g_list = []                            # Stores g(Z) across iterations
        self.e_list = []                            # Stores e(T, theta) across iterations
        self.log_e_tilde_list = []                  # Stores log_e_tilde(T, theta) across iterations
        self.max_log_likelihood = []                # Stores the maximum log-likelihood of particles across iterations

        self.S = args.n_particles                   # Stores the number of particles
        self.particles = []                         # Stores particle trees across iterations
        self.particle_log_likelihoods = []          # Stores particle log-likelihoods across iterations
        self.S_tensor = np.zeros((self.n_nodes, self.n_nodes, alphabet_size, alphabet_size))  # Stores the S tensor

        self.cal_e_list = []                        # Stores cal_e across iterations
        self.log_W_list = []                        # Stores log_W across iterations
        self.t_opts_list = []                       # Stores t_opts across iterations

        self.max_ll = np.nan
        self.mean_ll = np.nan
        self.std_ll = np.nan

        self.selected_particles = None
        self.selected_ll_weights = None
        self.best_epoch = None
        self.final_statistics = {}

        self.init_strategy = args.init_strategy                             # Stores tree initialization parameter
        self.samp_strategy = args.samp_strategy                             # Stores tree sampling parameter
        self.samp_strategy_eval = args.samp_strategy_eval
        self.branch_strategy = args.branch_strategy                         # Stores branch length sampling parameter

        self.explore_rate = args.slantis_explore_rate                       # Slantis explore rate (0 is default Slantis)
        self.csmc_tempering = args.samp_csmc_tempering                      # CSMC sampling tempering (1 corresponds to CSMC without tempering)
        self.n_csmc_trees = args.n_csmc_trees                               # N trees used in CSMC for sampling one particle
        self.csmc_merg_stategy = args.csmc_merg_stategy

        self.n_iwelbo_samples = args.n_iwelbo_samples
        self.n_iwelbo_runs = args.n_iwelbo_runs

        self.dataset_name = args.dataset                                    # Stores dataset name
        self.exp_name = get_experiment_setup_name(args)                     # Stores experiment name
        self.data_path = args.data_path                                     # Folder to store all data files
        self.result_path = args.result_path                                 # Folder to store all result files
        self.data_nex = self.data_path + self.dataset_name + ".nex"         # Sequence data in Nexus format
        self.data_fasta = self.data_path + self.dataset_name + ".fasta"     # Sequence data in Fasta format
        self.data_phylip = self.data_path + self.dataset_name + ".phylip"   # Sequence data in Phylip format
        self.true_tree_newick = self.data_path + self.dataset_name + ".nw"  # True tree in Newick format

        self.file_prefix = args.file_prefix

        # True tree, if data is synthetic
        self.true_tree_loglikelihood, self.true_tree = get_likelihood_from_newick_file(self.true_tree_newick, self.data)

        self.optimizer = args.optimizer

        # adam params
        self.m_dw, self.v_dw = 0, 0
        self.m_db, self.v_db = 0, 0
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8
        self.eta = args.ng_stepsize

    def nj_phyml_initialization(self):
        """ This function
            i) gets NJ tree from sequence data
            ii) runs PhyML to optimize branch lengths
            iii) returns k-best MST from the final weight matrix. """

        from data_utils import change_taxon_names

        logger = logging.getLogger('nj_phyml()')
        logger.info("NJ-based PhyML initialization started.")

        # Check if phylip file exists. Otherwise, create it
        orig_fname = self.data_phylip + "_orig"
        if not os.path.isfile(orig_fname):
            logger.warning("Phylip_orig file does not exist. Creating the file to: %s" % orig_fname)
            from Bio import AlignIO
            try:
                AlignIO.convert(self.data_nex, "nexus", open(orig_fname, "w"), "phylip", molecule_type="DNA")
            except:
                logger.info("Replicate truncated taxon names. Changing the names.")
                change_taxon_names(self.data_nex, "nexus", orig_fname, "phylip", prefix="taxon")

        if not os.path.isfile(self.data_phylip):
            logger.warning("Phylip file does not exist. Creating the file to: %s" % self.data_phylip)
            change_taxon_names(orig_fname, "phylip", self.data_phylip, "phylip", prefix="taxon")

        # Run PhyML on NJ tree to optimize the branch lengths
        cmd = "phyml -i " + self.data_phylip + " -m JC69 --r_seed " + str(self.seed) + " -o l -c 1"
        logger.info("Running PhyML.")
        logger.info("%s" % cmd)
        os.system(cmd)

        # Read optimized tree and clean the taxon names.
        phylip_out_fname = self.data_phylip + "_phyml_tree.txt"
        with open(phylip_out_fname, 'r') as file:
            nj_phyml_newick = file.read()
        nj_phyml_newick = nj_phyml_newick.replace('taxon', '')

        logger.debug("NJ-tree after optimization. \t%s" % nj_phyml_newick)
        tree_filename = self.result_path + '/' + self.model + '_nj_tree_phyml_after.png'
        save_tree_image(tree_filename, nj_phyml_newick, n_taxa=self.n_taxa)

        nj_tree = newick2nx(nj_phyml_newick, self.n_taxa)
        update_topology(nj_tree, self.root)

        if self.true_tree is not None:
            ete_res = ete_compare(nx2ete(nj_tree, self.root), nx2ete(self.true_tree, self.root))
            logger.info("NJ Tree. RF: %s \tmax RF: %s \tnorm RF: %s" %
                        (ete_res['rf'], ete_res['max_rf'], ete_res['norm_rf']))

        # Calculate loglikelihood of NJ tree
        up_table = self.phylo.compute_up_messages(nj_tree)
        nj_ll = compute_loglikelihood(up_table)
        logger.info("NJ-tree after optimization LL: \t%s" % nj_ll)

        # Calculate vertex marginals, t_opts and log_W
        down_table = self.phylo.compute_down_messages(nj_tree, up_table)
        marginals = self.phylo.compute_marginal_nodes(nj_tree, up_table, down_table)
        self.g = marginals
        t_opts = nx.floyd_warshall_numpy(nj_tree, np.arange(self.n_nodes),
                                         weight='t')  # Compute node to node distance matrix
        log_W, S = self.phylo.compute_edge_weights(self.n_nodes, self.g, t_opts, return_S=True)

        # Sample trees
        particle_trees = self.sample_trees(n_trees=self.S, log_W=log_W, t_opts=t_opts, phi=S)

        logger.info("S trees from NJ-tree is selected.")
        logger.debug("First out of S trees, after NJ optimization. \t%s" % nx_to_newick(particle_trees[0], self.root))
        return particle_trees

    def sample_trees(self, n_trees, log_W, t_opts, phi, return_q=False):
        if self.samp_strategy == "slantis":
            from slantis import sample_trees as sample_trees_slantis
            tree_list, q_list = sample_trees_slantis(log_W, t_opts, n_particles=n_trees, return_q=True, explore_rate=self.explore_rate)
        elif self.samp_strategy == "csmc":
            tree_list = []
            q_list = []
            for i in range(0, n_trees):
                t_opts = self.phylo.compute_branch_lengths(self.n_nodes, self.g, cal_e=None,
                                                              strategy=self.branch_strategy)
                log_W = self.phylo.compute_edge_weights(self.n_nodes, self.g, t_opts)
                tau, pi, log_Z = sample_trees_csmc(self.data, log_W, t_opts, K=self.n_csmc_trees, phi=phi,
                                                   beta=self.csmc_tempering, merger=self.csmc_merg_stategy)
                tree_list.append(tau)
                q_list.append(pi - log_Z)
        else:
            logging.error("Sampling method not found! %s" % self.samp_strategy)
            sys.exit()
        for tree in tree_list:
            update_topology(tree, self.root)

        if return_q:
            return tree_list, q_list
        else:
            return tree_list

    def initialize(self):
        logger = logging.getLogger('vaiphy.initialization')

        # Ancestral sequence initialization g(Z)
        # Uniform except leaves
        for i in range(self.n_taxa):
            self.g[i] = np.array([nuc_vec[c] for c in self.data[i]])
            self.g[i] /= self.g[i].sum(axis=-1, keepdims=True)

        # Tree Initialization
        # Neighbor-joining on observed data, optimize the branch lengths using PhyML and get k-best mst
        if self.init_strategy == "nj_phyml":
            logger.info("Initializing the trees by "
                        "i) NJ tree from data ii) optimize branch lengths iii) sample S trees.")
            init_trees = self.nj_phyml_initialization()
        # Random initialization
        else:  # init_strategy == "random"
            logger.info("Initializing the trees randomly.")
            init_trees = np.empty(self.S, dtype=np.object)
            for indx_s in range(self.S):
                init_trees[indx_s] = create_tree(self.n_taxa)

        # If NJ opt is done, all trees have the same branch-length matrix automatically
        if self.init_strategy == 'nj_phyml':
            t_init = np.zeros((self.n_nodes, self.n_nodes))
            for s in range(self.S):
                t_init += nx.floyd_warshall_numpy(init_trees[s], np.arange(self.n_nodes),
                                                  weight='t')  # Compute node to node distance matrix
            t_init /= self.S
        # If not, sample initial branch lengths from the prior distribution
        else:
            t_init = np.random.exponential(0.01, size=(self.n_nodes, self.n_nodes)) / 2
            t_init += t_init.T
            np.fill_diagonal(t_init, 0)

        # Initialize the tree weights uniformly
        e = np.ones(self.S) / self.S  # e(T, theta).
        log_e_tilde = np.log(np.ones(self.S) / self.S)  # Unnormalized log_e_tilde(T, theta)
        log_e = log_e_tilde - logsumexp(log_e_tilde)

        # TODO This part re-initializes g(Z) after the NJ-based tree initialization.
        # Remove the comments below lines to enable the re-initialization.
        # self.g = np.ones((self.root + 1, self.n_pos, alphabet_size)) * (1/alphabet_size)
        # Uniform except leaves
        # for i in range(self.n_taxa):
        #    self.g[i] = np.array([nuc_vec[c] for c in self.data[i]])

        return self.g, init_trees, t_init, e, log_e, log_e_tilde

    def get_cal_e(self, n_nodes, particle_trees=None, e_trees=None):
        """
        This method calculates the cal_e matrix of shape (n_nodes x n_nodes). Each element is in [0,1].
        The function uses the particle trees and their approximated probabilities to compute cal_e.
        :param n_nodes: Number of nodes in the tree. Int.
        :param particle_trees: A list of particle trees. Each tree is a NetworkX object.
        :param e_trees:  A list of particle tree probabilities. Each element is in [0,1], sums to 1.
        :return: cal_e: A (n_nodes x n_nodes) matrix of edge probabilities. Each element is in [0,1].
        """

        if particle_trees is None or e_trees is None:
            print("Error! Tree particle tree or e_trees parameter of get_cal_e is None!")
            sys.exit()

        cal_e = np.zeros((n_nodes, n_nodes))
        for i in range(len(particle_trees)):
            tree = particle_trees[i]

            for u, v in tree.edges:
                cal_e[u, v] += e_trees[i]
                cal_e[v, u] += e_trees[i]

        return cal_e

    def update_e_T_theta(self, log_W, t_opts, phi=None, eps=1e-1):
        """ Given the ancestral sequence dis5tribution g(Z), this function updates e(T, theta).
            Based on the bifurcation strategy, it modifies the sampled trees.
        """
        # Sample trees
        particle_trees, log_comp_q = self.sample_trees(n_trees=self.S, log_W=log_W, t_opts=t_opts, phi=phi, return_q=True)

        particles = []
        ll_weights = []
        log_w_list = []
        log_e_tilde = []

        for k, T_l in enumerate(particle_trees):
            loglik = compute_loglikelihood(self.phylo.compute_up_messages(T_l.copy()))
            particles.append(T_l)
            ll_weights.append(loglik)

            log_w = T_l.size(weight="weight")
            log_w_list.append(log_w)
            log_e_tilde.append(log_w)

        # Normalize log_e and e
        log_e = log_e_tilde - np.array(log_comp_q)  # logsumexp(log_e_tilde)
        e = np.exp(log_e) + eps
        # e /= np.exp(log_comp_q) + eps
        e /= np.sum(e)

        return particles, ll_weights, log_e_tilde, e, log_e

    def adam_optimizer(self, dw, indx_iter):
        ## momentum beta 1
        # *** weights *** #
        self.m_dw = self.beta1 * self.m_dw + (1 - self.beta1) * dw

        ## rms beta 2
        # *** weights *** #
        self.v_dw = self.beta2 * self.v_dw + (1 - self.beta2) * (dw ** 2)

        ## bias correction
        m_dw_corr = self.m_dw / (1 - self.beta1 ** (indx_iter + 1))
        v_dw_corr = self.v_dw / (1 - self.beta2 ** (indx_iter + 1))

        self.g = self.g_list[-1] - self.eta * (m_dw_corr / (np.sqrt(v_dw_corr) + self.epsilon))

    def get_iwelbo(self, log_W, t_opts, S, n_iwelbo_samples=10, n_iwelbo_runs=100):
        # the log-probability of the observations given the trees is computed using DP
        # the probability of sampling the map-estimate from the map-estimate is 1, i.e. q(Lambda) = 1
        # log_qtree: the unnormalized log-importance weights of each tree in the particle system
        # prior over branch lengths is Exp(10), where 10 is the expectation, i.e. rate = 0.1
        # the prior over trees is uniform, the number of possible trees are (2*n_leaves - 3)!!
        np.random.seed(self.seed)

        all_lls = []
        all_lbs = []
        # log_ptree = - np.sum(np.log(np.arange(3, self.n_nodes-1, 2)))  # TODO Note that this is not necessarily true since we have multifurcating trees
        n_internals = self.n_nodes - self.n_taxa
        log_ptree = -(self.n_nodes - 2) * np.log(n_internals) # + np.sum(np.log(np.arange(1, n_internals + 1)))
        for l in range(n_iwelbo_runs):
            log_ratio = np.zeros(n_iwelbo_samples)
            sampled_trees, log_qtree = self.sample_trees(n_iwelbo_samples, log_W, t_opts, phi=S, return_q=True)
            #sampled_trees, log_qtree = sample_trees_slantis(log_W, t_opts, n_particles=n_iwelbo_samples, return_q=True)
            # sampled_trees, log_qtree = self.sample_trees(n_iwelbo_samples, log_W, t_opts, return_q=True)

            for k, T in enumerate(sampled_trees):
                log_plambda = 0
                log_qlambda = 0
                for u, v in T.edges:
                    T[u][v]["weight"] = log_W[u, v]
                    T[u][v]["t"], log_qlambda_uv = sample_t_jc(S_ij=S[u, v], return_q=True)
                    log_qlambda += log_qlambda_uv
                    log_plambda += np.log(1 / 0.1) - 1 / 0.1 * T[u][v]["t"]
                up_table = self.phylo.compute_up_messages(T.copy())
                log_py_lambdatree = compute_loglikelihood(up_table.copy())
                all_lls.append(log_py_lambdatree)
                log_ratio[k] = log_py_lambdatree + log_plambda + log_ptree - log_qtree[k] - log_qlambda
            iwelbo = logsumexp(log_ratio - np.log(n_iwelbo_samples))
            all_lbs.append(iwelbo)

        all_lbs = np.array(all_lbs)
        all_lls = np.array(all_lls)
        mean_lb = np.mean(all_lbs)
        std_lb = np.std(all_lbs)
        max_ll = np.max(all_lls)
        mean_ll = np.mean(all_lls)
        std_ll = np.std(all_lls)
        return mean_lb, std_lb, max_ll, mean_ll, std_ll

    def get_iwelbo_conditional(self, log_W, t_opts, S, n_iwelbo_samples=10, n_iwelbo_runs=100):
        # the log-probability of the observations given the trees is computed using DP
        # the probability of sampling the map-estimate from the map-estimate is 1, i.e. q(Lambda) = 1
        # log_qtree: the unnormalized log-importance weights of each tree in the particle system
        # prior over branch lengths is Exp(10), where 10 is the expectation, i.e. rate = 0.1
        # the prior over trees is uniform, the number of possible trees are (2*n_leaves - 3)!!
        np.random.seed(self.seed)

        all_lls = []
        all_lbs = []
        # log_ptree = - np.sum(np.log(np.arange(3, self.n_nodes-1, 2)))  # TODO Note that this is not necessarily true since we have multifurcating trees
        log_ptree = 0
        L = 1
        for l in range(n_iwelbo_runs):
            log_ratio = np.zeros((n_iwelbo_samples, L))

            for k in range(n_iwelbo_samples):
                log_W, S = self.phylo.compute_edge_weights(self.n_nodes, self.g_list[-1], self.t_opts_list[-1],
                                                        distort_S=True, poi_distortion_rate=0, return_S=True)
                T, log_qtree = self.sample_trees(1, log_W, t_opts, return_q=True)
                T = T[0]
                for ell in range(L):
                    log_plambda = 0
                    log_qlambda = 0
                    for u, v in T.edges:
                        T[u][v]["weight"] = log_W[u, v]
                        T[u][v]["t"], log_qlambda_uv = sample_t_jc(S_ij=S[u, v], return_q=True)
                        log_qlambda += log_qlambda_uv
                        log_plambda += np.log(1 / 0.1) - 1 / 0.1 * T[u][v]["t"]
                    up_table = self.phylo.compute_up_messages(T.copy())
                    log_py_lambdatree = compute_loglikelihood(up_table.copy())
                    all_lls.append(log_py_lambdatree)
                    log_ratio[k, ell] = log_py_lambdatree + log_plambda + log_ptree - log_qtree - log_qlambda
            iwelbo = logsumexp(log_ratio - np.log(n_iwelbo_samples * L))
            all_lbs.append(iwelbo)

        all_lbs = np.array(all_lbs)
        all_lls = np.array(all_lls)
        mean_lb = np.mean(all_lbs)
        std_lb = np.std(all_lbs)
        max_ll = np.max(all_lls)
        mean_ll = np.mean(all_lls)
        std_ll = np.std(all_lls)
        return mean_lb, std_lb, max_ll, mean_ll, std_ll

    def get_miselbo(self, log_W, t_opts, S, n_iwelbo_samples=10, n_iwelbo_runs=100):
        # the log-probability of the observations given the trees is computed using DP
        # the probability of sampling the map-estimate from the map-estimate is 1, i.e. q(Lambda) = 1
        # log_qtree: the unnormalized log-importance weights of each tree in the particle system
        # prior over branch lengths is Exp(10), where 10 is the expectation, i.e. rate = 0.1
        # the prior over trees is uniform, the number of possible trees are (2*n_leaves - 3)!!
        np.random.seed(self.seed)

        all_lbs = []
        all_miselbos = []
        log_ptree = - np.sum(np.log(np.arange(3, self.n_nodes - 1, 2)))
        for l in range(n_iwelbo_runs):
            log_ratio_elbo = np.zeros(n_iwelbo_samples)
            log_ratio_miselbo = np.zeros(n_iwelbo_samples)
            sampled_trees = []
            log_pi = []
            log_Z = []
            log_q_b_list = []
            log_W_list = []
            for i in range(0, n_iwelbo_samples):
                log_q_b_mat = np.zeros_like(t_opts)
                t_opts = np.zeros_like(t_opts)
                for node1 in range(self.n_nodes):
                    for node2 in range(node1 + 1, self.n_nodes):
                        S_ij = self.phylo.approx_count(node1, node2, self.g)
                        t_opt, log_q_ = sample_t_jc(S_ij, return_q=True)
                        log_q_b_mat[node1, node2] = log_q_
                        t_opts[node1, node2] = t_opt
                        t_opts[node2, node1] = t_opt
                log_q_b_list.append(log_q_b_mat)
                log_W = self.phylo.compute_edge_weights(self.n_nodes, self.g, t_opts)
                log_W_list.append(log_W)
                tau, log_pi_i, log_Z_i = sample_trees_csmc(self.data, log_W, t_opts, K=self.n_csmc_trees, phi=S)
                update_topology(tau, self.root)
                sampled_trees.append(tau)
                log_pi.append(log_pi_i)
                log_Z.append(log_Z_i)
            for k, T in enumerate(sampled_trees):
                log_W = log_W_list[k]
                log_p_b = 0
                log_q_b = 0
                log_q_b_mat = log_q_b_list[k]
                log_pi_k = log_pi[k]
                for u, v in T.edges:
                    T[u][v]["weight"] = log_W[u, v]
                    log_p_b += np.log(1 / 0.1) - 1 / 0.1 * T[u][v]["t"]
                    log_q_b += log_q_b_mat[u, v]

                up_table = self.phylo.compute_up_messages(T.copy())
                log_py_lambdatree = compute_loglikelihood(up_table.copy())
                log_ratio_elbo[k] = log_py_lambdatree + log_p_b + log_ptree - log_q_b - (log_pi_k - log_Z[k])
                log_ratio_miselbo[k] = log_py_lambdatree + log_p_b + log_ptree - log_q_b - log_pi_k - logsumexp(-np.array(log_Z)) + np.log(len(sampled_trees))
            elbo = np.mean(log_ratio_elbo)
            miselbo = np.mean(log_ratio_miselbo)
            all_lbs.append(elbo)
            all_miselbos.append(miselbo)

        all_miselbos = np.array(all_miselbos)
        all_lbs = np.array(all_lbs)

        mean_mlb = np.mean(all_miselbos)
        std_mlb = np.mean(all_miselbos)
        mean_lb = np.mean(all_lbs)
        std_lb = np.std(all_lbs)
        return mean_mlb, std_mlb, mean_lb, std_lb

    def train(self, is_print=False):
        logger = logging.getLogger('vaiphy.train')
        logger.info("VaiPhy started.")
        t_start = time.time()

        # Initialization
        np.random.seed(self.seed)
        self.g, init_trees, t_opts, e, log_e, log_e_tilde = self.initialize()
        logger.info("Initialization is finished.")

        np.random.seed(self.seed)
        # Iteration 0
        current_particles = []          # Stores the particle trees temporarily at each iteration
        current_log_likelihoods = []    # Stores the log-likelihoods temporarily at each iteration
        for indx_s, tree in enumerate(init_trees):
            current_particles.append(tree)
            loglik = compute_loglikelihood(self.phylo.compute_up_messages(tree))
            current_log_likelihoods.append(loglik)
        log_W = None
        cal_e = self.get_cal_e(self.n_nodes, particle_trees=current_particles, e_trees=e)
        # Save particles and their log_likelihoods
        self.e_list.append(e)
        self.log_e_tilde_list.append(log_e_tilde)
        self.particles.append(current_particles)
        self.particle_log_likelihoods.append(current_log_likelihoods)
        self.max_log_likelihood.append(max(current_log_likelihoods))
        self.g_list.append(self.g)
        self.cal_e_list.append(cal_e)
        self.log_W_list.append(log_W)
        self.t_opts_list.append(t_opts)
        best_mean_lb = np.NINF
        lbs = []

        # VaiPhy loop
        for indx_iter in range(self.max_iter):
            logger.debug("Iteration: %s" % indx_iter)

            if is_print and (indx_iter % 20 == 0 or indx_iter < 3):
                logger.info("Iteration: %s \t" % indx_iter)
                best_particle_idx = np.argmax(current_log_likelihoods)
                logger.debug("Best particle's Newick: \t %s"
                             % nx_to_newick(current_particles[best_particle_idx], self.root))

                # Save tree image
                for s in range(min(self.S, 3)):
                    tree_filename = self.file_prefix + '_iter_' + str(indx_iter) + '_tree_' + str(s) + '.png'
                    save_tree_image(tree_filename, nx_to_newick(current_particles[s], self.root), n_taxa=self.n_taxa)

                calE_filename = self.file_prefix + '_iter_' + str(indx_iter) + '_calE.png'
                self.plot_cal_E(calE_filename, cal_e, iter=indx_iter)

            # Step 1. Calculate branch length matrix, using q(Z)
            # If it is the first iteration, use the actual tree branch lengths, rather than t_opts
            # if indx_iter != 0:
            #     t_opts, S = self.phylo.compute_branch_lengths(self.n_nodes, self.g, cal_e=cal_e,
            #                                               strategy=self.branch_strategy, return_S=True)
            # self.t_opts_list.append(t_opts)

            # Step 2. Calculate weight matrix, using q(Z) and branch lengths
            # log_W = self.phylo.compute_edge_weights(self.n_nodes, self.g, t_opts)
            # self.log_W_list.append(log_W)

            # Step 3. Calculate cal_e, using log_W or the particle trees & their normalized weights
            cal_e = self.get_cal_e(self.n_nodes, particle_trees=self.particles[-1], e_trees=self.e_list[-1])
            self.cal_e_list.append(cal_e)

            # Step 4. Update q(Z) with natural gradient, using cal_e and t_opts
            dw = self.phylo.compute_g(self.g, cal_e, t_opts)

            if self.optimizer == 'adam':
                self.adam_optimizer(dw, indx_iter)
            else:
                self.g = (1 - self.ng_stepsize) * self.g_list[-1] \
                          + self.ng_stepsize * dw
            self.g_list.append(self.g)

            # Step 5. Update q(T), using q(lambda) and q(Z)
            # TODO Remove t,w calculation?
            t_opts, S = self.phylo.compute_branch_lengths(self.n_nodes, self.g, cal_e=cal_e, strategy=self.branch_strategy, return_S=True)
            log_W = self.phylo.compute_edge_weights(self.n_nodes, self.g, t_opts)
            self.log_W_list.append(log_W)
            self.t_opts_list.append(t_opts)

            current_particles, current_log_likelihoods, log_e_tilde, e, log_e = self.update_e_T_theta(log_W=log_W,
                                                                                                      t_opts=t_opts, phi=S)

            self.particles.append(current_particles)
            self.particle_log_likelihoods.append(current_log_likelihoods)
            self.max_log_likelihood.append(max(current_log_likelihoods))
            self.e_list.append(e)

            if indx_iter % 5 == 0:
                utils.check_tree_diversity(self.particles[-1], logger)

            if indx_iter >= 0:
                if self.samp_strategy_eval == 'slantis':
                    mean_lb, *_ = self.get_iwelbo(log_W, t_opts, S, n_iwelbo_samples=self.n_iwelbo_samples,
                                                  n_iwelbo_runs=self.n_iwelbo_runs)
                    print("IWELBO: ", mean_lb)
                else:
                    mean_mlb, std_mlb, mean_lb, std_lb = self.get_miselbo(
                        log_W, t_opts, S, n_iwelbo_samples=self.n_iwelbo_samples, n_iwelbo_runs=self.n_iwelbo_runs)
                    print("MISELBO: ", mean_mlb)
                    print("ELBO: ", mean_lb)

                lbs.append(mean_lb)
                if mean_lb > best_mean_lb:
                    best_mean_lb = mean_lb
                    self.best_epoch = indx_iter
                    self.S_tensor = S
                    logger.info('Best lower bound: %s at iteration %s ' % (mean_lb, indx_iter))
        plt.plot(lbs)
        plt.ylabel('IWELBO')
        plt.xlabel('Epochs')
        plt.savefig('IWELBO_plot' + self.dataset_name)

        if self.best_epoch == 0:
            self.best_epoch = self.max_iter

        # TODO Maybe we should remove the below final update?
        # Get the final branch-lengths and edge-weights # TODO Do we need to recalculate cal_e as well?
        # if return_q = True and strategy = 'jc' then log_q_lambda_matrix is has non-zero entries, '
        # otherwise it's a zero matrix

        self.final_statistics['t_opts'], self.final_statistics['S'] = self.phylo.compute_branch_lengths(
            self.n_nodes, self.g, cal_e=cal_e, strategy=self.branch_strategy, return_S=True)
        self.final_statistics['log_W'] = self.phylo.compute_edge_weights(self.n_nodes, self.g, t_opts)

        # Save final values
        self.final_statistics['g'] = self.g
        self.final_statistics['cal_e'] = cal_e

        # Save only the values up until best idx
        self.g_list = self.g_list[:self.best_epoch + 1]
        self.log_W_list = self.log_W_list[:self.best_epoch + 1]
        self.cal_e_list = self.cal_e_list[:self.best_epoch + 1]
        self.t_opts_list = self.t_opts_list[:self.best_epoch + 1]
        self.particles = self.particles[:self.best_epoch + 1]
        self.particle_log_likelihoods = self.particle_log_likelihoods[:self.best_epoch + 1]
        self.max_log_likelihood = self.max_log_likelihood[:self.best_epoch + 1]

        filename = self.file_prefix + '_object.pkl'
        logger.info("Saving model object to: %s." % filename)
        save(self, filename)

        print("\nComputing Final IWELBO")
        mean_lb, std_lb, *_ = self.get_iwelbo(self.log_W_list[-1], self.t_opts_list[-1], self.S_tensor, n_iwelbo_samples=20, n_iwelbo_runs=5)
        print("Final IWELBO: ", mean_lb, "pm", std_lb)
        mean_lb, std_lb, *_ = self.get_miselbo(self.log_W_list[-1], self.t_opts_list[-1], self.S_tensor, n_iwelbo_samples=20, n_iwelbo_runs=5)
        print("MISELBO: ", mean_lb, "pm", std_lb)

        logger.info("VaiPhy is finished.")
        logger.info("Time taken by function train(): %s", str(time.time() - t_start))


    def analysis(self):
        logger = logging.getLogger('vaiphy.analysis')

        # Report the Edmonds tree
        self.report_edmonds_tree()

        # Pick the best particle's tree and return statistics
        final_particles = self.particles[-1]
        final_particle_loglikelihoods = self.particle_log_likelihoods[-1]
        idx = np.argmax(final_particle_loglikelihoods)
        selected_tree = final_particles[idx]
        selected_tree_ll = final_particle_loglikelihoods[idx]
        tree_filename = self.file_prefix + '_selected_tree.png'
        save_tree_image(tree_filename, nx_to_newick(selected_tree, self.root), n_taxa=self.n_taxa)
        logger.info("Selected tree log-likelihood: %s", str(selected_tree_ll))
        logger.info("Selected tree Newick: \t%s" % nx_to_newick(selected_tree, self.root))

        selected_tree_bifurcated = bifurcate_tree(selected_tree, self.n_taxa)
        selected_tree_bifurcated_loglik = compute_loglikelihood(self.phylo.compute_up_messages(selected_tree_bifurcated))
        tree_filename = self.file_prefix + '_selected_tree_bifurcated.png'
        save_tree_image(tree_filename, nx_to_newick(selected_tree_bifurcated, self.root), n_taxa=self.n_taxa)
        logger.info("Selected tree log-likelihood (bifurcated): %s", str(selected_tree_bifurcated_loglik))
        logger.info("Selected tree Newick (bifurcated): \t%s" % nx_to_newick(selected_tree_bifurcated, self.root))

        if self.true_tree_loglikelihood is not None:
            ete_res = ete_compare(nx2ete(selected_tree, self.root), nx2ete(self.true_tree, self.root))
            logger.info("Selected tree. RF: %s \tmax RF: %s \tnorm RF: %s" %
                        (ete_res['rf'], ete_res['max_rf'], ete_res['norm_rf']))
            ete_res = ete_compare(nx2ete(selected_tree_bifurcated, self.root), nx2ete(self.true_tree, self.root))
            logger.info("Selected tree (bifurcated). RF: %s \tmax RF: %s \tnorm RF: %s" %
                        (ete_res['rf'], ete_res['max_rf'], ete_res['norm_rf']))

            logger.info("True tree Newick: \t%s" % nx_to_newick(self.true_tree, self.root))

            print("\nTrue tree:")
            draw_dendrogram(self.true_tree, self.root)
        print("\nSelected tree: ")
        draw_dendrogram(selected_tree, self.root)
        print("\nSelected tree (bifurcated): ")
        draw_dendrogram(selected_tree_bifurcated, self.root)

        # Save consensus
        final_particles_bifurcated = []
        for tree in final_particles:
            final_particles_bifurcated.append(bifurcate_tree(tree, self.n_taxa))
        for cutoff in [0.5, 0.1]:
            fname = self.file_prefix + "_majority_consensus_cutoff_" + str(cutoff)

            consensus_newick = save_consensus_tree(fname, root=self.root,
                                                   tree_list=final_particles_bifurcated, cutoff=cutoff)
            logger.info("Consensus tree Newick (cutoff=%s): \t%s" % (cutoff, consensus_newick))
            consensus_tree = newick2nx(consensus_newick, self.n_taxa)
            update_topology(consensus_tree, self.root)
            loglik = compute_loglikelihood(self.phylo.compute_up_messages(consensus_tree))
            logger.info("Consensus tree's Log-Likelihood (cutoff=%s): \t %s" % (cutoff, loglik))
            if self.true_tree_loglikelihood is not None:
                ete_res = ete_compare(nx2ete(consensus_tree, self.root), nx2ete(self.true_tree, self.root))
                logger.info("Consensus tree (cutoff=%s). RF: %s \tmax RF: %s \tnorm RF: %s" %
                            (cutoff, ete_res['rf'], ete_res['max_rf'], ete_res['norm_rf']))

        if False:
            logger.info("Plotting started.")

            logger.info("Plotting particle with the maximum log-likelihood")
            plt.figure(figsize=(20, 10))
            plt.plot(range(len(self.max_log_likelihood)), self.max_log_likelihood)
            if self.true_tree_loglikelihood is not None:
                plt.plot(range(len(self.max_log_likelihood)),
                         np.ones(len(self.max_log_likelihood)) * self.true_tree_loglikelihood, 'r--')
            plt.title("Particle with the Maximum Log-likelihood.")
            plt.xlabel("Iterations")
            plt.savefig(self.file_prefix + '_max_ll.png')
            plt.close()

            logger.info("Plotting particle weights.")
            filename = self.file_prefix + '_weights.png'
            self.plot_particle_weights(filename, self.e_list)

            logger.info("Plotting ancestral sequence distributions.")
            filename = self.file_prefix + '_gZ'
            self.plot_ancestral_dist(filename, self.g_list[0:5])
            logger.info("Plotting finished.")

    def report_edmonds_tree(self):
        logger = logging.getLogger('vaiphy.edmonds()')
        edmonds_tree = get_k_best_mst(W=self.log_W_list[-1], k=1, t=self.t_opts_list[-1], alg="edmonds")[0]
        update_topology(edmonds_tree, self.root)
        loglik = compute_loglikelihood(self.phylo.compute_up_messages(edmonds_tree))
        logger.info("Edmonds tree's Log-Likelihood: \t %s" % loglik)
        logger.info("Edmonds tree's Newick: \t %s" % nx_to_newick(edmonds_tree, self.root))

        edmonds_tree_bifurcated = bifurcate_tree(edmonds_tree, self.n_taxa)
        loglik = compute_loglikelihood(self.phylo.compute_up_messages(edmonds_tree_bifurcated))
        logger.info("Edmonds tree's Log-Likelihood (bifurcated): \t %s" % loglik)
        logger.info("Edmonds tree's Newick (bifurcated): \t %s" % nx_to_newick(edmonds_tree_bifurcated, self.root))

        if self.true_tree_loglikelihood is not None:
            ete_res = ete_compare(nx2ete(edmonds_tree, self.root), nx2ete(self.true_tree, self.root))
            logger.info("Edmonds tree. RF: %s \tmax RF: %s \tnorm RF: %s"
                        % (ete_res['rf'], ete_res['max_rf'], ete_res['norm_rf']))
            ete_res = ete_compare(nx2ete(edmonds_tree_bifurcated, self.root), nx2ete(self.true_tree, self.root))
            logger.info("Edmonds tree (bifurcated). RF: %s \tmax RF: %s \tnorm RF: %s"
                        % (ete_res['rf'], ete_res['max_rf'], ete_res['norm_rf']))
        tree_filename = self.file_prefix + '_final_edmonds_tree.png'
        save_tree_image(tree_filename, nx_to_newick(edmonds_tree, self.root), n_taxa=self.n_taxa)
        tree_filename = self.file_prefix + '_final_edmonds_tree_bifurcated.png'
        save_tree_image(tree_filename, nx_to_newick(edmonds_tree_bifurcated, self.root), n_taxa=self.n_taxa)

    def plot_cal_E(self, filename, cal_e_matrix, iter=None):
        plt.imshow(cal_e_matrix, interpolation='nearest', vmin=0, vmax=1)
        plt.xlabel("Nodes")
        plt.ylabel("Nodes")

        title_str = "CalE Matrix"
        if iter is not None:
            title_str += " at iteration " + str(iter)
        plt.title(title_str)
        plt.savefig(filename)
        plt.close()

    def plot_particle_weights(self, filename, e_list):
        num_iter = len(e_list)
        S = len(e_list[0])

        fig, ax1 = plt.subplots(figsize=(30, 10))

        color = 'tab:gray'
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.set_ylabel("Particle Index", color=color)
        ax1.set_xlabel("Iterations")
        ax1.set_title("The Weights of Particles e(T, theta) & ELBO Across Iterations")

        for it in range(num_iter):
            ax1.vlines(it, 0, S + 1, colors=color, linestyles="solid", alpha=0.1)
            for s in range(S):
                ax1.scatter(it, s + 1, s=300 * e_list[it][s], alpha=0.4, c=color)
        plt.savefig(filename)
        plt.close()

    def plot_ancestral_dist(self, filename, g_list, min_threshold=0):
        num_iter = len(g_list)
        num_vertex, _, _ = g_list[0].shape

        for it in range(num_iter):
            plt.figure(figsize=(20, np.ceil(num_vertex / 2)))
            plt.title("Ancestral Sequence Distribution g(Z) at Iteration %d" % it)
            for i in range(num_vertex):
                for j in range(2):
                    v = 2 * i + j
                    if v >= num_vertex:
                        break
                    # print("Vertex: ", v, "\ti,j: ", i, j)
                    ax_v = plt.subplot2grid((num_vertex, 2), (i, j))
                    ax_v.imshow(g_list[it][v, :, :].T, interpolation="nearest", aspect="auto", vmin=min_threshold,
                                vmax=1)

            new_filename = filename + "_iter_" + str(it) + ".png"
            plt.savefig(new_filename)
            plt.close()
