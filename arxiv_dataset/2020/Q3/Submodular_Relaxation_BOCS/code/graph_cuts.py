# Author: Aryan Deshwal

# File containing the implementation of the submodular relaxation based AFO

from graph_tool.all import *
import numpy as np
class GraphCuts():
    def __init__(self, n_vars, input_lambda, alpha, logger):
        # n_vars : input dimensions 
        # input_lambda: lambda parameter (if applicable)
        # alpha : parameter vector sampled from the surrogate model
        # logger : logger object
        self.n_vars = n_vars
        self.alpha = alpha
        self.b = alpha[1:n_vars+1] + input_lambda
        self.upper_indices = np.triu_indices(self.n_vars, 1)
        self.A = np.zeros((self.n_vars, self.n_vars))
        self.A[self.upper_indices] = self.alpha[n_vars+1:]
        self.A_pos = np.copy(self.A)
        self.A_neg = np.copy(self.A)
        self.A_pos[self.A_pos <= 0] = 0 
        self.A_neg[self.A_neg >= 0] = 0 
        self.one_vector = np.ones(self.n_vars)
        self.logger = logger
        self.logger.debug("submodular relaxation optimization")

    def initialize_gamma(self, random_set):
        # random_set : sample initial gamma randomly or the strategy proposed in https://bit.ly/3aA2ijn
        if random_set:  
            self.gamma = np.random.uniform(size=(self.n_vars, self.n_vars))
        else:
            self.gamma = np.outer(self.one_vector, self.one_vector)/2

    def update_gamma(self, previous_res, t):
        temp = np.outer(self.one_vector, self.one_vector) - np.outer(previous_res, self.one_vector) - \
                                                             np.outer(self.one_vector, previous_res)
        self.S_t = np.multiply(self.A_pos, temp)
        self.gamma = self.gamma - (self.S_t/np.sqrt(t+1))
        self.gamma[self.gamma < 0] = 0.
        self.gamma[self.gamma > 1] = 1.

    def create_relaxed_objective(self):
        # separating out the non-negative and non-positive alphas in the quadratic term
        # non-positive alphas are submodular -> direcly representable as graphs
        # non-negative alphas needs to be relaxed to a linear from 
        self.hadamard_product = np.multiply(self.A_pos, self.gamma)
        self.h_gamma = np.dot(self.one_vector, self.hadamard_product) + \
                             np.dot(self.hadamard_product, self.one_vector) + self.b
        self.extra_sub = np.dot(np.dot(self.one_vector, self.hadamard_product), self.one_vector)



    def get_solution_min_cut(self, random_set, boykov, max_t):
        # add n_vars + 2 (source and target) vertices to the graph
        t = max_t
        best_value = np.inf 
        best_input = []
        all_inputs = []
        all_vals = []
        self.initialize_gamma(random_set)
        count = 0
        for n_iter in range(t):
            self.create_relaxed_objective()
            self.graph = Graph()
            self.graph.add_vertex(self.n_vars + 2)
            self.source = 0
            self.target = self.n_vars + 1
            # add the capacity property
            cap = self.graph.new_edge_property("double")
            # add edges for first order terms
            for i in range(self.n_vars):
                if (self.h_gamma[i] > 0):
                    e = self.graph.add_edge(self.source, i+1)
                    cap[e] = self.h_gamma[i]
                    # print("edge: ",self.source, i+1, cap[e])
                elif (self.h_gamma[i] < 0):
                    e = self.graph.add_edge(i+1, self.target)
                    cap[e] = -1*self.h_gamma[i]
                    # print("edge: ", i+1, self.target, cap[e])

            # add edges for second order terms
            # add (v_i, v_j) -alpha_{ij} capacity
            # add (v_j, t) -alpha_{ij} capacity
            for i in range(self.n_vars):
                for j in range(i+1, self.n_vars):
                    if self.A_neg[i][j] != 0:
                        count += 1
                        e = self.graph.add_edge(j+1, self.target)
                        cap[e] = -1*self.A_neg[i][j]
                        e = self.graph.add_edge(i+1, j+1)
                        cap[e] = -1*self.A_neg[i][j]
            if boykov:
                res = boykov_kolmogorov_max_flow(self.graph, self.source, self.target, cap)
            else:
                res = push_relabel_max_flow(self.graph, self.source, self.target, cap)
            source_based_partition = min_st_cut(self.graph, self.source, cap, res).get_array()[1:-1].astype(np.float)
            partition = np.asarray([not(x) for x in source_based_partition])
            # res.a = cap.a - res.a  # the actual flow
            new_val = self.alpha[0] + np.dot(partition.T, np.dot(self.A, partition)) + np.dot(self.b, partition)
            if (new_val < best_value):
                best_value = new_val
                best_input = partition
            self.logger.debug("new_val: %s"%new_val)
            all_inputs.append(partition)
            all_vals.append(new_val)
            if not random_set:
                self.update_gamma(partition, n_iter)
            else:
                self.initialize_gamma(True)
        return best_input, self.extra_sub, np.asarray(all_inputs), np.asarray(all_vals)
# -- END OF FILE --








