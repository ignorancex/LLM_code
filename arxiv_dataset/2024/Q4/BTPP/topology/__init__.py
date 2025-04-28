import torch
import numpy as np
import math

from .topo_static import RingGraph, GridGraph, StarGraph, ExponentialGraph, FullyConnectedGraph
from .topo_dyn import OnePeerExponentialGraph, CECA1PGraph, CECA2pGraph
from .topo_prob import *
from .topo_pp import PushPullTreeGraph
from .topo_finite_sum import *
from .topo_neighbor import *


def get_topology(args):
    """
    generate the weighted matrix for the topology
    """
    if args.topo == 'ring':
        return RingGraph(args.nodes)
    elif args.topo == 'grid':
        return GridGraph(args.nodes)
    elif args.topo == 'star':
        return StarGraph(args.nodes)
    elif args.topo == 'exponential':
        return ExponentialGraph(args.nodes)
    elif args.topo == 'fully_connected':
        return FullyConnectedGraph(args.nodes)
    elif args.topo == 'OnePeerExp':
        return OnePeerExponentialGraph(args.nodes)
    elif args.topo == 'base_k':
        return BaseGraph(args.nodes, 2).w_list
    elif args.topo == 'pp':
        return PushPullTreeGraph(args.nodes)
    elif args.topo == 'pp_ring':
        matrix_R = RingGraph(args.nodes, 1)
        matrix_C = RingGraph(args.nodes, 2)
        return matrix_R, matrix_C
    elif args.topo == 'ceca-1p':
        return CECA1PGraph(args.nodes)
    elif args.topo == 'ceca-2p':
        return CECA2pGraph(args.nodes)
    
def get_prob_topology(args, ite):
    if args.topo == 'ODEquiDyn':
        M = math.log2(args.nodes) 
        M = int(M) - 1
        _, Ms = DEquiStatic(args.nodes, seed=ite, M=M)
        A = ODEquiDyn(args.nodes, Ms, eta = 0.5, rng = np.random.default_rng(ite))
        return A
    elif args.topo == 'OUEquiDyn':
        M = math.log2(args.nodes) 
        M = int(M) - 1
        _, Ms = UEquiStatic(args.nodes, seed=ite, M=M)
        A = OUEquiDyn(args.nodes, Ms, eta = 0.5, rng = np.random.default_rng(ite))
        return A

def get_relay_neighbor(args, rank):
    if args.topo == 'relay_chain':
        return RelayChainNeighbor(args.nodes, rank)
    elif args.topo == 'relay_binarytree':
        return RelayBinaryTreeNeighbor(args.nodes, rank)


