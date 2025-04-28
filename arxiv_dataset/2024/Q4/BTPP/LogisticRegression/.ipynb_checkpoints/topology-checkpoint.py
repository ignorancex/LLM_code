"""
RingGraph
GridGraph
ExponentialGraph
OnePeerExponentialGraph
CECA1PGraph
CECA2PGraph
StarGraph
BaseGraph
FullyConnectedGraph
PushPullTreeGraph
"""

import numpy as np
import math 
from itertools import chain

import torch
import copy
import sympy


def RingGraph(size:int, connect_style: int = 0):
    """
    Generate ring structure of graph (uniliteral).
    Args:
        size: number of nodes
        connect_style: 0: double connected ring graph, with weight 1/3
                       1: left connected graph with 1/2
                       2: right connected graph with 1/2
    """
    assert size > 0
    if size == 1:
        return np.array([[1.0]])
    if size == 2:
        return np.array([[0.5, 0.5], [0.5, 0.5]])

    x = np.zeros(size)
    x[0] = 0.5
    if connect_style == 0:
        x[0] = 1/3.
        x[1] = 1/3.
        x[-1] = 1/3.
    elif connect_style == 1:
        x[-1] = 0.5
    elif connect_style == 2:
        x[1] = 0.5
    else:
        raise ValueError("Connect_style has to be int between 0 and 2")
    
    topo = np.empty((size, size))
    for i in range(size):
        topo[i] = np.roll(x, i)

    return topo

def GridGraph(size:int):
    """
    Generate grid structure of graph (uniliteral).
    Args:
        size: number of nodes
    """
    n = size
    n_sqrt = int(np.sqrt(n))
    
    neighbors_list = {}
    degrees = {}
    for k in range(n):
        nb_list = [k]
        
        # case 1: for vertex
        if k == 0:
            nb_list.append(k+1)
            nb_list.append(k+n_sqrt)
        elif k == n_sqrt-1:
            nb_list.append(k-1)
            nb_list.append(k+n_sqrt)
        elif k == n-n_sqrt:
            nb_list.append(k+1)
            nb_list.append(k-n_sqrt)
        elif k == n-1:
            nb_list.append(k-1)
            nb_list.append(k-n_sqrt)
        elif k > 0 and k < n_sqrt-1:
            nb_list.append(k-1)
            nb_list.append(k+1)
            nb_list.append(k+n_sqrt)
        elif k > n-n_sqrt and k < n-1:
            nb_list.append(k-1)
            nb_list.append(k+1)
            nb_list.append(k-n_sqrt)
        elif k%n_sqrt == 0:
            nb_list.append(k-n_sqrt)
            nb_list.append(k+1)
            nb_list.append(k+n_sqrt)
        elif k%n_sqrt == n_sqrt-1:
            nb_list.append(k-n_sqrt)
            nb_list.append(k-1)
            nb_list.append(k+n_sqrt)
        else:
            nb_list.append(k-1)
            nb_list.append(k+1)
            nb_list.append(k-n_sqrt)
            nb_list.append(k+n_sqrt)
            
        neighbors_list[k] = nb_list
        degrees[k] = len(nb_list)
            
    A = np.zeros((n, n))
    for k in range(n):
        for l in neighbors_list[k]:
            if l == k:
                continue            
            A[k,l] = 1./max(degrees[k], degrees[l]) # metropolis_rule
        
    for k in range(n):
        A[k,k] = 1. - np.sum(A[:,k])
        
    return A

def isPowerOf(x, base):
    assert isinstance(base, int), "`base` has to be an integer"
    assert base > 1, "`base` has to be an integer greater than 1"
    assert x > 0
    if (base** int(math.log(x, base))) == x:
        return True
    return False

def ExponentialGraph(size:int, base:int = 2):
    """
    Generate exponential graph such that each points only
    connected to a point such that the index differnce is power of `base`. (Default is 2) 
    Args:
        size: number of nodes
        base: index difference, the larger, the sparser.
    """
    x = [1.0]
    for i in range(1,size):
        if isPowerOf(i, base):
            x.append(1.0)
        else:
            x.append(0.0)
    x = np.array(x)
    x /= x.sum()
    topo = np.empty((size, size))
    for i in range(size):
        topo[i] = np.roll(x,i)

    return topo

def OnePeerExponentialGraph(size:int):
    """
    Generate exponential graph such that each points only
    connected to a point such that the index differnce is power of `base`. (Default is 2)
    Args:
        size: number of nodes
    Returns:
        a list of matrices representing sub-graphs
    """
    n = size
    subgraph_list = []
    
    for i in range(1, n):    
        incidenceMat = np.zeros((n,n))
        base_row = np.zeros(n)
        base_row[0] = 1
        if i&(i-1) == 0:
            base_row[i] = 1
            for i in range(n):
                incidenceMat[i,:] = np.roll(base_row,i) 
            subgraph_list.append(incidenceMat/2)
    
    return subgraph_list

def CECA1PGraph(n):
    """
    Generate CECA-1-Port graph
    Args:
        n: number of nodes
    """
    tau = int(math.ceil(math.log(n, 2.0))) 
    bi = bin(n - 1)
    n_ex = 0
    Ws = np.zeros((tau, 2 * n, 2 * n))
    Wgs = np.zeros((tau, 2 * n, 2 * n))
    for k in range(tau):
        d = int(bi[2 + k])
        P = np.zeros((n, n))
        if d == 1:
            for j in range(n):
                if j % 2 == 1:
                    i = (j + 2 * n_ex + 1) % n
                    P[i, j] = 1
                    P[j, i] = 1
            
            Ws[k, :n, :n] = 0.5 * np.eye(n) + 0.5 * P
            Ws[k, :n, n:] = 0
            Ws[k, n:, :n] = (n_ex + 1) * P / (2 * n_ex + 1)
            Ws[k, n:, n:] = n_ex * np.eye(n) / (2 * n_ex + 1)
            Wgs[k, :n, :n] = 0.5 * np.eye(n) + 0.5 * P
            Wgs[k, :n, n:] = 0
            Wgs[k, n:, :n] = (n_ex + 1) * P / (2 * n_ex + 1) + n_ex * np.eye(n) / (2 * n_ex + 1)
            Wgs[k, n:, n:] = 0
        
        if d == 0:
            for j in range(n):
                if j % 2 == 1:
                    i = (j + 2 * n_ex + 1) % n
                    P[i, j] = 1
                    P[j, i] = 1
            
            Ws[k, :n, :n] = (n_ex + 1) * np.eye(n) / (2 * n_ex + 1)
            Ws[k, :n, n:] = n_ex * P / (2 * n_ex + 1)
            Ws[k, n:, :n] = 0
            Ws[k, n:, n:] = 0.5 * np.eye(n) + 0.5 * P
            Wgs[k, :n, :n] = 0
            Wgs[k, :n, n:] = (n_ex + 1) * np.eye(n) / (2 * n_ex + 1) + n_ex * P / (2 * n_ex + 1)
            Wgs[k, n:, :n] = 0
            Wgs[k, n:, n:] =  0.5 * np.eye(n) + 0.5 * P
        n_ex = 2 * n_ex + d
    return Ws, Wgs

def CECA2pGraph(n):
    """
    Generate CECA-2-Port graph
    Args:
        n: number of nodes
    """
    tau = int(math.ceil(math.log(n, 2.0))) 
    bi = bin(n - 1)
    n_ex = 0
    Ws = np.zeros((tau, 2 * n, 2 * n))
    Wgs = np.zeros((tau, 2 * n, 2 * n))
    for k in range(tau):
        d = int(bi[2 + k])
        P = np.zeros((n, n))
        if d == 1:
            for j in range(n):
                i = (j + n_ex + 1) % n
                P[i, j] = 1
            
            Ws[k, :n, :n] = 0.5 * np.eye(n) + 0.5 * P
            Ws[k, :n, n:] = 0
            Ws[k, n:, :n] = (n_ex + 1) * P / (2 * n_ex + 1)
            Ws[k, n:, n:] = n_ex * np.eye(n) / (2 * n_ex + 1)
            Wgs[k, :n, :n] = 0.5 * np.eye(n) + 0.5 * P
            Wgs[k, :n, n:] = 0
            Wgs[k, n:, :n] = (n_ex + 1) * P / (2 * n_ex + 1) + n_ex * np.eye(n) / (2 * n_ex + 1)
            Wgs[k, n:, n:] = 0
        
        if d == 0:
            for j in range(n):
                i = (j + n_ex) % n
                P[i, j] = 1
            
            Ws[k, :n, :n] = (n_ex + 1) * np.eye(n) / (2 * n_ex + 1)
            Ws[k, :n, n:] = n_ex * P / (2 * n_ex + 1)
            Ws[k, n:, :n] = 0
            Ws[k, n:, n:] = 0.5 * np.eye(n) + 0.5 * P
            Wgs[k, :n, :n] = 0
            Wgs[k, :n, n:] = (n_ex + 1) * np.eye(n) / (2 * n_ex + 1) + n_ex * P / (2 * n_ex + 1)
            Wgs[k, n:, :n] = 0
            Wgs[k, n:, n:] =  0.5 * np.eye(n) + 0.5 * P
        n_ex = 2 * n_ex + d
    return Ws, Wgs

def StarGraph(size:int, center_rank:int = 0):
    """
    Generate star graph such that all other ranks are connected with the center_rank (default is 0)
    Args:
        size: number of nodes
        center_rank: center rank
    """
    assert size > 0
    topo = np.zeros((size, size))
    for i in range(size):
        topo[i,i] = 1 - 1/size
        topo[center_rank, i] =  1/size
        topo[i, center_rank] = 1/size

    return topo

def FullyConnectedGraph(size:int):
    """
    Generate fully connected graph
    Args:
        size: number of nodes
    """
    assert size > 0
    x = np.array([1/size]*size)
    topo = np.empty((size, size))
    for i in range(size):
        topo[i] = np.roll(x,i)

    return topo

def PushPullTreeGraph(n, B=2):
    """
    Generate push pull B-ary tree graph
    Args:
        n: number of nodes
        B: number of branchs of parent nodes
    return: a list of matrices representing sub-graphs
    """
    graph_list = []
    R = np.zeros((n,n))
    N = np.ceil((n+1)/B).astype(int)
    for i in range(N):
        if i == 1:
            R[:i+B, 0] = 1.
        else:
            R[B*(i-1)+ 1:B*i + 1, i-1] = 1
    R[B*(N-1)+1:, N-1] = 1
    C = R.T
    graph_list.append(R)
    graph_list.append(C)

    return graph_list


def DEquiStatic(n, seed=0, eps=None, p=None, M=None):
    """A function that generates static topology for directed graphs satisfying
        Pr( ||Proj(W)||_2 < eps ) >= 1 - p  
    Args:
        n: number of nodes
        seed: an integer used as the random seed
        eps: the upper bound of l2 norm
        p: the probability that the l2 norm is bigger than eps
        M: conmunnication cost. If M is not given, M is calculated from eps and p. 
    Returns:
        K: a numpy array that specifies the communication topology.
        As: a sequence of basis index
    """
    if M == None:
        M = int(8 * math.log(2 * n / p) / 3 / eps**2)
    # generating M graphs
    np.random.seed(seed)
    As = np.random.choice(np.arange(1, n), size=M, replace=True)
    Ws = np.zeros((n,n))
    for a in As:
        W = np.zeros((n,n))
        for i in range(1, n + 1):
            j =  (i + a) % n
            if j == 0: j = n
            W[i-1, j-1] = (n - 1) / n
            W[i-1, i-1] = 1 / n
        Ws += W
    K = Ws / M
    #assert is_doubly_stochastic(K)
    return K, As

def UEquiStatic(n, seed=0, eps=None, p=None, M=None):
    """A function that generates static topology for undirected graphs satisfying
        Pr( ||Proj(W)||_2 < eps ) >= 1 - p  
    Args:
        n: number of nodes
        seed: an integer used as the random seed
        eps: the upper bound of l2 norm
        p: the probability that the l2 norm is bigger than eps
        M: conmunnication cost. If M is not given, M is calculated from eps and p. 
    Returns:
        K: a numpy array that specifies the communication topology.
        As: a sequence of basis index
    """
    if M == None:
        M = int(8 * math.log(2 * n / p) / 3 / eps**2)
    # generating M graphs
    np.random.seed(seed)
    As = np.random.choice(np.arange(1, n), size=M, replace=True)
    Ws = np.zeros((n,n))
    for a in As:
        W = np.zeros((n,n))
        for i in range(1, n + 1):
            j =  (i + a) % n
            if j == 0: j = n
            W[i-1, j-1] = (n - 1) / n
            W[i-1, i-1] = 1 / n
        Ws += W + W.T
    K = Ws / M / 2
    #assert is_doubly_stochastic(K)
    #assert is_symmetric(K)
    return K, As

def ODEquiDyn(n, Ms, eta=1, rng=np.random.default_rng(0)):
    """A function that generates onepeer topology from D-EquiStatic.
    Args:
        n: number of nodes
        Ms: a sequence of basis index
        ets: a hyperparameter for adjusting weights, 0< eta <=1
        rng: random number generator
    Returns:
        W: a numpy array that specifies the communication topology.
    """
    p = rng.choice(Ms, size=1)
    W = np.zeros((n,n))
    for i in range(1, n + 1):
        j = (i + p) % n
        if j == 0: j = n
        W[i-1, j-1] = (n - 1) / n
        W[i-1, i-1] = 1 / n
    W = (1 - eta) * np.eye(n) + eta * W
    #assert is_doubly_stochastic(W)
    return W

def ODEquiDynComplete(n, eta=1, rng=np.random.default_rng(0)):
    """A function that generates onepeer topology from D-EquiStatic with M=n-1.
    Args:
        n: number of nodes
        ets: a hyperparameter for adjusting weights, 0< eta <=1
        rng: random number generator
    Returns:
        W: a numpy array that specifies the communication topology.
    """
    p = rng.choice(np.arange(1, n), size=1)
    W = np.zeros((n,n))
    for i in range(1, n + 1):
        j = (i + p) % n
        if j == 0: j = n
        W[i-1, j-1] = (n - 1) / n
        W[i-1, i-1] = 1 / n
    W = (1 - eta) * np.eye(n) + eta * W
    #assert is_doubly_stochastic(W)
    return W

def OUEquiDyn(n, Ms, eta=1, rng=np.random.default_rng(0)):
    """A function that generates onepeer topology from U-EquiStatic.
    Args:
        n: number of nodes
        Ms: a sequence of basis index
        ets: a hyperparameter for adjusting weights, 0< eta <=1
        rng: random number generator
    Returns:
        W: a numpy array that specifies the communication topology.
    """
    p = rng.choice(Ms, size=1)
    s = rng.choice(np.arange(1, n + 1), size=1)
    W = np.zeros((n,n))
    z = np.zeros(n)
    for i in chain(range(int(s), n+1), range(1, int(s))):
        j = (i + p) % n
        if j == 0: j = n
        if z[i-1] == 0 and z[j-1] == 0:
            W[i-1, j-1] = 1
            W[j-1, i-1] = 1
            z[i-1] = 1
            z[j-1] = 1
    for i in range(n):
        if z[i] == 0:
            W[i, i] = 1
    W = np.eye(n) / n + (n - 1) * W / n
    W = (1 - eta) * np.eye(n) + eta * W
    #assert is_doubly_stochastic(W)
    #assert is_symmetric(W)
    return W

def OUEquiDynComplete(n, eta=1, rng=np.random.default_rng(0)):
    """A function that generates onepeer topology from U-EquiStatic with M=n-1.
    Args:
        n: number of nodes
        ets: a hyperparameter for adjusting weights, 0< eta <=1
        rng: random number generator
    Returns:
        W: a numpy array that specifies the communication topology.
    """
    p = rng.choice(np.arange(1, n), size=1)
    s = rng.choice(np.arange(1, n + 1), size=1)
    W = np.zeros((n,n))
    z = np.zeros(n)
    for i in chain(range(int(s), n+1), range(1, int(s))):
        j = (i + p) % n
        if j == 0: j = n
        if z[i-1] == 0 and z[j-1] == 0:
            W[i-1, j-1] = 1
            W[j-1, i-1] = 1
            z[i-1] = 1
            z[j-1] = 1
    for i in range(n):
        if z[i] == 0:
            W[i, i] = 1
    W = np.eye(n) / n + (n - 1) * W / n
    W = (1 - eta) * np.eye(n) + eta * W
    #assert is_doubly_stochastic(W)
    #assert is_symmetric(W)
    return W



class DynamicGraph():
    def __init__(self, w_list):
        """
        Parameter
        --------
        w_list (list of torch.tensor):
            list of mixing matrix
        """
        self.w_list = w_list
        self.n_nodes = w_list[0].size()[0]
        self.length = len(w_list)
        self.itr = 0
        
    def get_in_neighbors(self, i):
        """
        Parameter
        ----------
        i (int):
            a node index
        Return
        ----------
            dictionary of (neighbors's index: weight of the edge (i,j))
        """
        w = self.w_list[self.itr%self.length]        

        return {idx.item(): w[idx, i].item() for idx in torch.nonzero(w[:,i])}

    def get_out_neighbors(self, i):
        """
        Parameter
        ----------
        i (int):
            a node index
        Return
        ----------
            dictionary of (neighbors's index: weight of the edge (i,j))
        """
        w = self.w_list[self.itr%self.length]        
        
        return {idx.item(): w[i,idx].item() for idx in torch.nonzero(w[i])}

    
    def get_neighbors(self, i):
        in_neighbors = self.get_in_neighbors(i)
        out_neighbors = self.get_out_neighbors(i)
        self.itr += 1
        return in_neighbors, out_neighbors
        
    
    def get_w(self):
        w = self.w_list[self.itr%self.length]        
        self.itr += 1
        return w
    
class HyperHyperCube(DynamicGraph):
    def __init__(self, n_nodes, seed=0, max_degree=1):
        self.state = np.random.RandomState(seed)
        self.max_degree = max_degree
        
        if n_nodes == 1:
            super().__init__([torch.eye(1)])
        else:
            if list(sympy.factorint(n_nodes))[-1] > max_degree+1:
                print(f"Can not construct {max_degree}-peer graphs")
        
            node_list = list(range(n_nodes))
            factors_list = self.split_node(node_list, n_nodes)
            #print(factors_list)
            super().__init__(self.construct(node_list, factors_list, n_nodes))
    
    def construct(self, node_list, factors_list, n_nodes):
        w_list = []
        for k in range(len(factors_list)):
            #print(factors_list)
            
            w = torch.zeros((n_nodes, n_nodes))
            b = torch.zeros(n_nodes)
            
            for i_idx in range(len(node_list)):
                for nk in range(1, factors_list[k]):
                    
                    i = node_list[i_idx]
                    j = int(i + np.prod(factors_list[:k]) * nk) % n_nodes
                    
                    if b[i] < factors_list[k]-1 and b[j] < factors_list[k]-1:
                        #print("g", i, j, b[i], b[j])
                        b[i] += 1
                        b[j] += 1
                        
                        w[i, j], w[j, i] = 1/factors_list[k], 1/factors_list[k]
                        w[i, i], w[j, j] = 1/factors_list[k], 1/factors_list[k]

            w_list.append(w)
       
        return w_list
                    

    def split_node(self, node_list, n_nodes):
        factors_list = []
        rest = n_nodes
        
        for factor in reversed(range(2, self.max_degree+2)):
            while rest % factor == 0:
                factors_list.append(factor)
                rest = int(rest / factor)

                if rest == 1:
                    break

        factors_list.reverse()
        return factors_list

    
class SimpleBaseGraph(DynamicGraph):
    def __init__(self, n_nodes, max_degree=1, seed=0, inner_edges=True):
        self.state = np.random.RandomState(seed)
        self.inner_edges = inner_edges
        self.max_degree = max_degree
        self.n_nodes = n_nodes

        super().__init__(self.construct())

    def construct(self):
        node_list_list, n_nodes_list = self.split_nodes()
        node_list_list_list = self.split_nodes2(node_list_list)
        L = len(node_list_list)
        
        if self.n_nodes == 1:
            return [torch.eye(1)]
        elif max(list(sympy.factorint(self.n_nodes))) <= self.max_degree + 1:
            return HyperHyperCube(self.n_nodes, max_degree=self.max_degree).w_list
        
        # construct k-peer HyperHyperCube
        hyperhyper_cubes = [HyperHyperCube(len(node_list_list[i]), max_degree=self.max_degree) for i in range(L)]        
        hyperhyper_cubes2 = [HyperHyperCube(len(node_list_list_list[i][0]), max_degree=self.max_degree) for i in range(L)]
        max_length_of_hyper = len(hyperhyper_cubes[0].w_list)

        b = torch.zeros(L)
        true_b = torch.tensor([len(hyperhyper_cube.w_list) for hyperhyper_cube in hyperhyper_cubes2])
        
        w_list = []
        m = -1
        while True:
            m += 1
            w = torch.zeros((self.n_nodes, self.n_nodes))
            isolated_nodes = None
            all_isolated_nodes = None
            
            for l in reversed(range(L)):
                
                if m < max_length_of_hyper:
                    length = len(hyperhyper_cubes[l].w_list)
                    w += self.extend(hyperhyper_cubes[l].w_list[m % length], node_list_list[l])
                    
                elif m < max_length_of_hyper + l:
                    if isolated_nodes is None:
                        isolated_nodes = copy.deepcopy(node_list_list_list[m - max_length_of_hyper])
                        all_isolated_nodes = [node for nodes in isolated_nodes for node in nodes]
                        
                    for i in node_list_list[l]:
                        a_l = len(isolated_nodes)
                        
                        for k in range(a_l):
                            j = isolated_nodes[k].pop(-1)
                            all_isolated_nodes.remove(j)
                            w[i, j] = n_nodes_list[m - max_length_of_hyper] / sum(n_nodes_list[m - max_length_of_hyper:]) / a_l
                            w[j, i] = n_nodes_list[m - max_length_of_hyper] / sum(n_nodes_list[m - max_length_of_hyper:]) / a_l

                            w[j, j] = 1 - w[i, j]
                        w[i, i] = 1 - n_nodes_list[m - max_length_of_hyper] / sum(n_nodes_list[m - max_length_of_hyper:])
                            
                elif m == max_length_of_hyper + l and l != L-1:
                    while len(all_isolated_nodes) > 1 and self.inner_edges:
                        sampled_nodes = all_isolated_nodes[:min(self.max_degree+1,len(all_isolated_nodes))]

                        for node_id in sampled_nodes:
                            all_isolated_nodes.remove(node_id)
                        
                        for i in sampled_nodes:
                            for j in sampled_nodes:
                                w[i, j] = 1 / len(sampled_nodes)
                                w[j, i] = 1 / len(sampled_nodes)
                                w[i, i] = 1 / len(sampled_nodes)
                                w[j, j] = 1 / len(sampled_nodes) 
            
                else:
                    if n_nodes_list[l] < self.max_degree+1:
                        length = len(hyperhyper_cubes[l].w_list)
                        w += self.extend(hyperhyper_cubes[l].w_list[int(b[l] % length)], node_list_list[l])
                    else:
                        a_l = len(node_list_list_list[l])
                        
                        for k in range(a_l):
                            length = len(hyperhyper_cubes2[l].w_list)
                            w += self.extend(hyperhyper_cubes2[l].w_list[int(b[l] % length)], node_list_list_list[l][k])
                        
                    b[l] += 1

            # add self-loop
            for i in range(self.n_nodes):
                if w[i, i] == 0:
                    w[i,i] = 1.0
            w_list.append(w)

            #if (b >= true_b).all():
            #    break
            if b[0] == len(hyperhyper_cubes2[0].w_list):
                break
            
        return w_list
            
    def diag(self, X, Y):
        new_W = torch.zeros((X.size()[0] + Y.size()[0], X.size()[0] + Y.size()[0]))
        new_W[0:X.size()[0], 0:X.size()[0]] = X
        new_W[X.size()[0]:, X.size()[0]:] = Y
        return new_W


    def extend(self, w, node_list):
        new_w = torch.zeros((self.n_nodes, self.n_nodes))
        for i in range(len(node_list)):
            for j in range(len(node_list)):
                new_w[node_list[i], node_list[j]] = w[i, j]
        return new_w

    def split_nodes(self):
        factor = (self.max_degree + 1)**int(math.log(self.n_nodes, self.max_degree+1))
        n_nodes_list = []
        
        while sum(n_nodes_list) != self.n_nodes:

            rest = self.n_nodes - sum(n_nodes_list)
            
            if rest >= factor:
                n_nodes_list.append((rest // factor) * factor)
            factor = int(factor/(self.max_degree  + 1))
        node_list = list(range(self.n_nodes))
        node_list_list = []
        for i in range(len(n_nodes_list)):
            node_list_list.append(node_list[sum(n_nodes_list[:i]):sum(n_nodes_list[:i+1])])

        return node_list_list, n_nodes_list

    
    def split_nodes2(self, node_list_list):
        """
        len(node_list) can be written as a_l * (max_degree + 1)^{p_l} where al \in \{1, 2, \cdots, k\}.
        """

        node_list_list_list = []
        
        for node_list in node_list_list:
            n_nodes = len(node_list)
            power = math.gcd(n_nodes, (self.max_degree+1) ** int(math.log(n_nodes, self.max_degree+1)))
            rest = int(n_nodes / power)

            node_list_list_list.append([])
            for i in range(rest):
                node_list_list_list[-1].append(node_list[i*power:(i+1)*power])
                
        return node_list_list_list
    

class BaseGraph(DynamicGraph):
    def __init__(self, n_nodes, max_degree=1, seed=0, inner_edges=True):
        self.state = np.random.RandomState(seed)
        self.inner_edges = inner_edges
        self.max_degree = max_degree
        self.n_nodes = n_nodes
        self.seed = seed
        
        super().__init__(self.construct())

    def construct(self):
        node_list_list1, node_list_list2, n_power, n_rest = self.split_nodes()

        simple_adics = [SimpleBaseGraph(len(node_list_list1[i]), max_degree=self.max_degree) for i in range(n_power)]
        hyper_cubes = [HyperHyperCube(len(node_list_list2[i]), max_degree=self.max_degree) for i in range(n_rest)]

        # check which is better
        g = SimpleBaseGraph(self.n_nodes, max_degree=self.max_degree, seed=self.seed, inner_edges=self.inner_edges)
        if len(g.w_list) < len(simple_adics[0].w_list) + len(hyper_cubes[0].w_list):
            return g.w_list
        
        
        w_list = []
        for m in range(len(simple_adics[0].w_list)):
            w = torch.zeros((self.n_nodes, self.n_nodes))
            
            for l in range(n_power):
                w += self.extend(simple_adics[l].w_list[m], node_list_list1[l])
            w_list.append(w)
            
        for m in range(len(hyper_cubes[0].w_list)):
            w = torch.zeros((self.n_nodes, self.n_nodes))
            
            for l in range(n_rest):
                w += self.extend(hyper_cubes[l].w_list[m], node_list_list2[l])
            w_list.append(w)

        return w_list
    
        
    def diag(self, X, Y):
        new_W = torch.zeros((X.size()[0] + Y.size()[0], X.size()[0] + Y.size()[0]))
        new_W[0:X.size()[0], 0:X.size()[0]] = X
        new_W[X.size()[0]:, X.size()[0]:] = Y
        return new_W


    def extend(self, w, node_list):
        new_w = torch.zeros((self.n_nodes, self.n_nodes))
        for i in range(len(node_list)):
            for j in range(len(node_list)):
                new_w[node_list[i], node_list[j]] = w[i, j]
        return new_w
    
    
    def split_nodes(self):
        factors = [n**int(math.log(self.n_nodes, n)) for n in range(2, self.max_degree+2)]
        factor = np.prod(factors) 
        n_power = math.gcd(self.n_nodes, factor)
        n_rest = int(self.n_nodes / n_power)

        node_list = list(range(self.n_nodes))
        node_list_list1 = []
        for i in range(n_power):
            node_list_list1.append(node_list[n_rest*i:n_rest*(i+1)])

        node_list_list2 = [[] for _ in range(n_rest)]
        for i in range(n_power):
            for j in range(n_rest):
                node_list_list2[j].append(node_list_list1[i][j])
            
        return node_list_list1, node_list_list2, n_power, n_rest

    
    def get_neighbors(self, i):
        in_neighbors = self.get_in_neighbors(i)
        out_neighbors = self.get_out_neighbors(i)
        self.itr += 1

        #if self.itr % len(self.w_list) == 0:
        #    self.w_list = self.shuffle_node_index(self.w_list, self.n_nodes)

        return in_neighbors, out_neighbors        

def AverageTestGraph(n):
    """
    Average models for fair testing
    Args:
        n: number of nodes
    """
    graph = np.zeros((n,n))
    graph[0, :] = 1/n
    return graph

def RelayChainNeighbor(n, rank):
    """
    show the neighborhood ranks in the chain graph given specific rank
    Args:
        n: number of nodes
        rank: rank of the node
    Returns:
        a list of ranks that are in the neighborhood of the input rank.
    """
    neighbor = []
    if rank == 0:
        neighbor.append(1)
    elif rank == n-1:
        neighbor.append(n-2)
    else:
        neighbor.append(rank-1)
        neighbor.append(rank+1)
    return neighbor

def RelayBinaryTreeNeighbor(n, rank, B = 2):
    """
    show the neighborhood ranks in the binary tree graph given specific rank
    Args:
        n: number of nodes
        rank: rank of the node
        B: number of branchs of parent nodes
    Returns:
        a list of ranks that are in the neighborhood of the input rank.
    """
    neighbor = []
    # add the child nodes
    for i in range(B):
        if B*rank + i < n - 1:
            neighbor.append(B*rank + i + 1)
    # add the parent node
    if rank > 0:
        neighbor.append((rank-1)//B)
    return neighbor