import math
import random
from scipy.sparse import csr_matrix
from tqdm import tqdm
from collections import defaultdict, deque
import time
import concurrent.futures
import numpy as np


class RWSampler:
    def __init__(self, adj_train, feat_full=None, sample_rate=0.01, walk_length=2, core_rate=0.5):
        self.fire_probs = {}
        self.per_nodes = {}
        self.sample_rate = sample_rate
        self.core_rate = core_rate
        self.walk_length = walk_length
        self.adj_train = adj_train
        self.degrees = np.array(self.adj_train.sum(1)).flatten()
        self.feat_full = feat_full if feat_full is not None else np.ones(adj_train.shape[0])
        self.N_cc = {}
        self.N_pc = {}
        self.N_pp = {}
        self.d_th = 0
        self.p_weight = {}


    def evaluate_network(self):

        t1 = time.time()
        V_org = np.arange(self.adj_train.shape[0])

        max_edge_count = 0
        edge_count_dict = defaultdict(int)
        row, col = self.adj_train.nonzero()
        edges = list(zip(row, col))
        edges_sampled = random.sample(edges,50000) if len(edges)> 50000 else edges
        for u, w in edges_sampled:
            u_degree = self.degrees[u]
            w_degree = self.degrees[w]
            for d in range(w_degree, u_degree):
                edge_count_dict[d] += 1
                if edge_count_dict[d] > max_edge_count:
                    max_edge_count = edge_count_dict[d]
                    self.d_th = d


        self.per_nodes = [v for v in V_org if self.degrees[v] <= self.d_th and self.degrees[v] != 0]

        for v in V_org:
            if self.degrees[v] != 0:
                neighbors = self.adj_train.indices[self.adj_train.indptr[v]:self.adj_train.indptr[v + 1]]

                if self.degrees[v] <= self.d_th:

                    Nc_p = [u for u in neighbors if self.degrees[u] > self.d_th]

                    prob = [math.sqrt(self.degrees[u]+1) * np.linalg.norm(self.feat_full[u])
                            for u in neighbors if self.degrees[u] > self.d_th]
                    self.fire_probs[v] = prob
                    self.N_pc[v] = Nc_p

                    Np_p = [u for u in
                            neighbors if self.degrees[u] <= self.d_th and self.degrees[u] != 0]
                    weight = [np.linalg.norm(self.feat_full[u]) / math.sqrt(self.degrees[u]+1) for u in neighbors
                              if self.degrees[u] <= self.d_th and self.degrees[u] != 0]
                    self.p_weight[v] = weight
                    self.N_pp[v] = Np_p
        t2 = time.time()
        return

    def sample_RW(self):

        n = int(np.count_nonzero(self.degrees) * self.sample_rate)
        node_index = set()

        while len(node_index) < n:
            v = self.per_nodes[random.randint(0, len(self.per_nodes) - 1)]
            if v not in node_index:
                node_index.add(v)

                if self.N_pc[v] and self.core_rate != 1:
                    core_temp = set(random.choices(self.N_pc[v], weights=self.fire_probs[v],
                                                   k=math.ceil(len(self.N_pc[v]) * self.core_rate)))

                elif self.N_pc[v] and self.core_rate == 1:
                    core_temp = set(self.N_pc[v])
                else:
                    core_temp = {}

                node_index.update(core_temp)

                for _ in range(self.walk_length):
                    if not self.N_pp[v]:
                        break
                    v = random.choices(self.N_pp[v], weights=self.p_weight[v], k=1)[0]
                    if v not in node_index:
                        node_index.add(v)

                        if self.N_pc[v] and self.core_rate != 1:
                            core_temp = set(random.choices(self.N_pc[v], weights=self.fire_probs[v],
                                                           k=math.ceil(len(self.N_pc[v]) * self.core_rate)))

                        elif self.N_pc[v] and self.core_rate == 1:
                            core_temp = set(self.N_pc[v])
                        else:
                            core_temp = {}

                        node_index.update(core_temp)

        sub_nodes = np.array(list(node_index))
        sub_nodes = np.sort(sub_nodes)
        sub_adj = self.adj_train[sub_nodes, :][:, sub_nodes]

        return sub_nodes, sub_adj

    def sampling(self):

        G_sampled = self.sample_RW()

        return G_sampled

    def sample_process(self, per_num):
        sub_graphs = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            with tqdm(total=per_num, desc="Progress") as pbar:
                for i in range(per_num):
                    futures.append(executor.submit(self.sampling))

                for future in concurrent.futures.as_completed(futures):
                    sub_graphs.append(future.result())
                    pbar.update(1)
        return sub_graphs

class FFSampler:
    def __init__(self, adj_train, feat_full, sample_rate=0.01, p=0.5, core_rate=0.5):
        self.candidates = None
        self.per_nodes = None
        self.sample_rate = sample_rate
        self.adj_train = adj_train
        self.degrees = np.array(self.adj_train.sum(1)).flatten()
        self.core_rate = core_rate
        self.feat_full = feat_full if feat_full is not None else np.ones(adj_train.shape[0])
        self.N_cc = {}
        self.N_pc = {}
        self.N_pp = {}
        self.d_th = 0
        self.p = p
        self.fire_probs = {}


    def evaluate_network(self):
        t1 = time.time()
        V_org = np.arange(self.adj_train.shape[0])
        max_edge_count = 0
        edge_count_dict = defaultdict(int)
        row, col = self.adj_train.nonzero()
        edges = list(zip(row, col))
        edges_sampled = random.sample(edges,50000) if len(edges)> 50000 else edges
        for u, w in edges_sampled:
            u_degree = self.degrees[u]
            w_degree = self.degrees[w]
            for d in range(w_degree, u_degree):
                edge_count_dict[d] += 1
                if edge_count_dict[d] > max_edge_count:
                    max_edge_count = edge_count_dict[d]
                    self.d_th = d

        for v in V_org:
            if self.degrees[v] != 0:
                neighbors = self.adj_train.indices[self.adj_train.indptr[v]:self.adj_train.indptr[v + 1]]

                if self.degrees[v] <= self.d_th:

                    Nc_p = [u for u in neighbors if self.degrees[u] > self.d_th]

                    prob = [math.sqrt(self.degrees[u]+1) * np.linalg.norm(self.feat_full[u])
                            for u in neighbors if self.degrees[u] > self.d_th]
                    self.fire_probs[v] = prob
                    self.N_pc[v] = Nc_p

                    Np_p = [(u, np.linalg.norm(self.feat_full[u]) / math.sqrt(self.degrees[u]+1)) for u in
                            neighbors if self.degrees[u] <= self.d_th and self.degrees[u] != 0]

                    self.N_pp[v] = Np_p

        self.per_nodes = {v for v in V_org if self.degrees[v] <= self.d_th and self.degrees[v] != 0}
        self.candidates = self.per_nodes.copy()
        t2 = time.time()
        return

    def sample_FF(self):
        Q = set()
        fifo = deque()

        n = int(np.count_nonzero(self.degrees) * self.sample_rate)
        if len(self.candidates) < n:
            self.candidates = self.per_nodes.copy()
        temp = self.candidates.copy()

        w = list(temp)[random.randint(0, len(temp) - 1)]
        fifo.append(w)
        Q.add(w)
        node_index = set()

        while len(node_index) < n and temp:
            if not Q:
                w = list(temp)[random.randint(0, len(temp) - 1)]
                Q.add(w)
                fifo.append(w)
            v = fifo.pop()
            Q.discard(v)
            if v not in node_index:
                node_index.add(v)

                if self.N_pc[v] and self.core_rate != 1:
                    core_temp = set(random.choices(self.N_pc[v], weights=self.fire_probs[v],
                                                   k=math.ceil(len(self.N_pc[v]) * self.core_rate)))
                elif self.N_pc[v] and self.core_rate == 1:
                    core_temp = set(self.N_pc[v])
                else:
                    core_temp = {}

                node_index.update(core_temp)

                remaining_nodes = []
                weights = []
                for u, p in self.N_pp[v]:
                    if u not in Q and u not in node_index:
                        remaining_nodes.append(u)
                        weights.append(p)

                temp.discard(v)
                self.candidates.discard(v)
                if remaining_nodes:
                    fraction = np.random.geometric(p=self.p)
                    to_add = set(random.choices(remaining_nodes, weights=weights, k=min(fraction, len(remaining_nodes))))
                    fifo.extend(to_add)
                    Q.update(to_add)

        sub_nodes = np.array(list(node_index))
        sub_nodes = np.sort(sub_nodes)
        sub_adj = self.adj_train[sub_nodes, :][:, sub_nodes]

        return sub_nodes, sub_adj

    def sampling(self):
        G_sampled = self.sample_FF()
        return G_sampled

    def sample_process(self, per_num):
        sub_graphs = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            with tqdm(total=per_num, desc="Progress") as pbar:
                for i in range(per_num):
                    futures.append(executor.submit(self.sampling))

                for future in concurrent.futures.as_completed(futures):
                    sub_graphs.append(future.result())
                    pbar.update(1)
        return sub_graphs
