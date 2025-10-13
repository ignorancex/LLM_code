from HIS.utils import *
from HIS.HISsampler import *
from HIS.model.inits import *
import numpy as np
import time


def _coo_scipy2torch(adj):
    """
    convert a scipy sparse COO matrix to torch
    """
    values = adj.data
    indices = np.vstack((adj.row, adj.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    return torch.sparse_coo_tensor(i, v, torch.Size(adj.shape))


class Minibatch:
    """
    Provides minibatches for the trainer or evaluator. This class is responsible for
    calling the proper graph sampler and estimating normalization coefficients.
    """

    def __init__(self, adj_full_norm, adj_train, role, train_params, feat_full=np.zeros(0), cpu_eval=False):
        """
        Inputs:
            adj_full_norm       scipy CSR, adj matrix for the full graph (row-normalized)
            adj_train           scipy CSR, adj matrix for the traing graph. Since we are
                                under transductive setting, for any edge in this adj,
                                both end points must be training nodes.
            role                dict, key 'tr' -> list of training node IDs;
                                      key 'va' -> list of validation node IDs;
                                      key 'te' -> list of test node IDs.
            train_params        dict, additional parameters related to training. e.g.,
                                how many subgraphs we want to get to estimate the norm
                                coefficients.
            cpu_eval            bool, whether we want to run full-batch evaluation
                                on the CPU.

        Outputs:
            None
        """

        self.feat_full = feat_full
        self.sampling_time = 0
        self.use_cuda = (args_global.gpu >= 0)
        if cpu_eval:
            self.use_cuda = False

        self.node_train = np.array(role['tr'])
        self.node_val = np.array(role['va'])
        self.node_test = np.array(role['te'])
        self.adj_full_norm = _coo_scipy2torch(adj_full_norm.tocoo())

        self.adj_train = adj_train
        self.adj_train, self.i = removing_multiple_and_selfloop_edges(self.adj_train)
        self.model = train_params['model']
        if self.use_cuda:

            self.adj_full_norm = self.adj_full_norm.cuda()


        self.sub_graphs = []
        self.batch_num = -1

        self.norm_loss_train = np.zeros(self.adj_train.shape[0])

        self.norm_loss_test = np.zeros(self.adj_full_norm.shape[0])
        _denom = len(self.node_train) + len(self.node_val) + len(self.node_test)
        self.norm_loss_test[self.node_train] = 1. / _denom
        self.norm_loss_test[self.node_val] = 1. / _denom
        self.norm_loss_test[self.node_test] = 1. / _denom
        self.norm_loss_test = torch.from_numpy(self.norm_loss_test.astype(np.float32))
        if self.use_cuda:
            self.norm_loss_test = self.norm_loss_test.cuda()
        self.norm_aggr_train = np.zeros(self.adj_train.size)

        self.sample_coverage = train_params['sample_coverage']
        self.deg_train = np.array(self.adj_train.sum(1)).flatten()
        self.num_nodes = np.count_nonzero(self.deg_train)


    def set_sampler(self, train_phases):
        """
        Inputs:
            train_phases       dict, config / params for the graph sampler

        Outputs:
            None
        """
        self.sample_rate = train_phases['sample_rate']
        self.method_sample = train_phases['sampler']
        if self.method_sample == 'rw':
            self.graph_sampler = RWSampler(self.adj_train, self.feat_full ,self.sample_rate,
                                           int(train_phases['depth']), train_phases['core_rate'])
        elif self.method_sample == 'ff':
            self.graph_sampler = FFSampler(self.adj_train, self.feat_full, self.sample_rate, train_phases['p'], train_phases['core_rate'])
        else:
            raise NotImplementedError

        self.graph_sampler.evaluate_network()

        self.norm_loss_train = np.zeros(self.adj_train.shape[0])

        while True:
            self.per_sample_num = self.sample_coverage * 100
            self.par_graph_sample(True)
            break
        print()

        for sub_graph in self.sub_graphs:
            self.norm_loss_train[sub_graph[0]] += 1
        num_G = len(self.sub_graphs)

        self.norm_loss_train[np.where(self.norm_loss_train == 0)[0]] = 0.1
        self.norm_loss_train[self.node_val] = 0
        self.norm_loss_train[self.node_test] = 0
        self.norm_loss_train[self.node_train] = num_G / self.norm_loss_train[self.node_train] / self.node_train.size
        self.norm_loss_train = torch.from_numpy(self.norm_loss_train.astype(np.float32))
        if self.use_cuda:
            self.norm_loss_train = self.norm_loss_train.cuda()

    def par_graph_sample(self, Flag=False):
        """
        Perform graph sampling in parallel. A wrapper function for graph_samplers.py
        """
        t0 = time.time()
        sampled_subgraphs = self.graph_sampler.sample_process(self.per_sample_num)
        t1 = time.time()
        print('sampling {} subgraphs:   time = {:.3f} sec'.format(self.per_sample_num, t1 - t0), end="\r")
        self.sampling_time += t1 - t0
        self.sub_graphs.extend(sampled_subgraphs)


    def one_batch(self, mode='train'):
        """
        Generate one minibatch for trainer. In the 'train' mode, one minibatch corresponds
        to one subgraph of the training graph. In the 'val' or 'test' mode, one batch
        corresponds to the full graph (i.e., full-batch rather than minibatch evaluation
        for validation / test sets).

        Inputs:
            mode                str, can be 'train', 'val', 'test' or 'valtest'

        Outputs:
            node_subgraph       np array, IDs of the subgraph / full graph nodes
            adj                 scipy CSR, adj matrix of the subgraph / full graph
            norm_loss           np array, loss normalization coefficients. In 'val' or
                                'test' modes, we don't need to normalize, and so the values
                                in this array are all 1.
        """
        if mode in ['val', 'test', 'valtest']:
            self.node_subgraph = np.arange(self.adj_full_norm.shape[0])
            adj = self.adj_full_norm

        else:
            assert mode == 'train'
            if len(self.sub_graphs) == 0:
                self.par_graph_sample()
                print()

            self.node_subgraph, adj= self.sub_graphs.pop()

            self.size_subgraph = len(self.node_subgraph)


            adj = adj_norm(adj, deg=self.deg_train[self.node_subgraph], model=self.model)
            adj = _coo_scipy2torch(adj.tocoo())
            if self.use_cuda:
                adj = adj.cuda()
            self.batch_num += 1
        norm_loss = self.norm_loss_test if mode in ['val', 'test', 'valtest'] else self.norm_loss_train
        norm_loss = norm_loss[self.node_subgraph]
        return self.node_subgraph, adj, norm_loss, self.i

    def num_training_batches(self):
        return math.ceil(1 / float(self.sample_rate))

    def shuffle(self):
        self.node_train = np.random.permutation(self.node_train)
        self.batch_num = -1

    def end(self):
        return self.batch_num + 1 >= 100
