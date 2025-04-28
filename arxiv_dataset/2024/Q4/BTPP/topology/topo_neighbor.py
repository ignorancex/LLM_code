import numpy as np
import math 
from itertools import chain

import torch
import copy
import sympy

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