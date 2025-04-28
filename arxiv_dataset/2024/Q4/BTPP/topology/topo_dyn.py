import numpy as np
import math 
from itertools import chain

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
