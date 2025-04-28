import numpy as np
import math 
from itertools import chain



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

def RingGraph(size:int, connect_style: int = 0):
    pass