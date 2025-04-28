import numpy as np
import math 
from itertools import chain


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