import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def reset_mediators(W,action_index):
    """
    Reset mediators if it has higher topological order than A or lower order than Y.

    Args:
    W: weights of DAG
    action_index: index of treatment/event

    Returns:
    Gx: New weighted DAG with potential moderators X and unit weights
    """
    # reset mediators if it has higher topological order than A or lower order than Y.
    ordered_vertices = list(nx.topological_sort(nx.DiGraph(W)))
    d = W.shape[0]
    j = 2*action_index + 1
    while j < d - 1:
        if  ordered_vertices.index(j) < ordered_vertices.index(action_index):
            W[j, 1:(d - 1)] = 0
        if  ordered_vertices.index(j) > ordered_vertices.index(d - 1):
            W[1:(d - 1), j] = 0
        j = j + 1
    return W

def add_X_to_graph(W,simple=True,p=1,noint=False,moderator=False,seed=42,
                  zero_chance=0.2):
    """Adds p-dimensional X to graph

    Args:
    W: weights of DAG
    simple: Flag for scenario 1; true if scenario 1, false otherwise
    p: number of potential moderators
    noint: if true, then no interaction will be added to the graph
    moderator: if true, then moderator will not directly affect the treatment
    seed: seed for the randomly generated weights for X
    zero_chance: likelihood the weights of X will be zero

    Returns:
    Gx: New weighted DAG with X and unit weights
    """
    np.random.seed(seed)
    d = W.shape[0]
    W = np.hstack((np.zeros([d,p]),W)) # Potential moderators have no parents
    # Make sure potential moderators are independent
    if simple: # not mediated
        # Get rid of all edges other than X -> A and X -> Y
        W = np.vstack((np.zeros([p,d+p]),W))
        # make sure at least one potential moderator has the treatment as a descendent
        rand = np.random.choice((-1,0,1),p)
        while sum(abs(rand)) == 0:
            rand = np.random.choice((-1,0,1),p)
        W[0:p,p] = rand
        W[0:p,-1] = np.random.choice((-1,0,1),p)
    else:
        Ones = np.ones([p,d+p])
        Ones[np.random.rand(p, d+p) < (1-zero_chance)/2] *= -1
        Ones[np.random.rand(p, d+p) < zero_chance] *= 0
        W = np.vstack((Ones,W))
        # Get rid of edges between X
        W[0:p,0:p] = 0
        if moderator: W[0:p,p] = 0 # remove potential moderator impact on A

    # add potential moderator/treatment interaction
    temp = np.zeros([2*p+d,2*p+d])
    temp[:(p+1),:(p+1)] = W[:(p+1),:(p+1)] # upper left sqaure
    temp[:(p+1),(2*p+1):] = W[:(p+1),(p+1):] # upper right rect
    if not noint:
        if simple:
            Ones = np.ones([p,1])
            Ones[np.random.rand(p, 1) < (1-zero_chance)/2] *= -1
            Ones[np.random.rand(p, 1) < zero_chance] *= 0
            temp[(p+1):(2*p+1),-1] = np.squeeze(Ones) # gamma_{XA} vector
        else:
            Ones = np.ones([p,d-1])
            Ones[np.random.rand(p, d-1) < (1-zero_chance)/2] *= -1
            Ones[np.random.rand(p, d-1) < zero_chance] *= 0
            temp[(p+1):(2*p+1),(2*p+1):] = Ones # B_{XA} matrix and gamma_{XA} vector
    temp[(2*p+1):,(2*p+1):] = W[(p+1):,(p+1):] # lower right sqaure

    return reset_mediators(temp,p)

def add_XM_to_graph(W,p=1,s=6,seed=42,zero_chance=0.8):
    
    np.random.seed(seed)
    d = W.shape[0]
    temp = np.zeros([s*p+d,s*p+d])
    temp[:(2*p+s+1),:(2*p+s+1)] = W[:(2*p+s+1),:(2*p+s+1)] # upper left sqaure
    temp[:(2*p+s+1),-1] = W[:(2*p+s+1),-1] # upper right rect
    temp[-1,:(2*p+s+1)] = W[-1, :(2*p+s+1)] # lower left rect
    
    # random rect
    Ones = np.ones([s*p,1])
    Ones[np.random.rand(s*p, 1) < (1-zero_chance)/2] *= -1
    Ones[np.random.rand(s*p, 1) < zero_chance] *= 0
    temp[(2*p+s+1):-1,-1] = np.squeeze(Ones) # gamma_{XM} vector

    return temp

def simulate_dag(s: int = 6, stype: str = 'simple', p: int = 1, degree: float = 2,
                 w_range: tuple = (1.0, 1.0), seed: int = 1, xseed: int = 42, noint=False,
                 moderator=False, zero_chance = 0.2, XM=False) -> nx.DiGraph:
    """
    Simulate random DAG with an expected degree by Erdos-Renyi model.
        
    Args:
    s: number of mediators
    stype: one of ('simple','parallel','interacted')
    p: number of potential moderators
    degree: expected node degree, in + out
    w_range: weight range +/- (low, high)
    seed: seed for random functions
    xseed: seed for random functions related to adding X to the generated graph
    noint: if true, then no interaction will be added to the graph
    moderator: if true, then moderator will not directly affect the treatment
    zero_chance: likelihood the weights of X will be zero
    
        
    Returns:
    G: weighted DAG
    W: weights for G without X and interactions
    W_X: weights for G with X and interactions
    """

    np.random.seed(seed)
    
    d = s+2 # number of nodes - moderator
    if stype == 'simple':
        W = np.zeros((d,d))
        W[0,-1] = np.random.choice((-1,1),1) # effect of A on Y
    elif stype == 'parallel':
        W = np.zeros((d,d))
        W[0,1:d-1] = np.random.choice((-1,0,1),d-2) # effect of A on M
        W[0,-1] = np.random.choice((-1,1),1) # effect of A on Y
        W[1:d-1,-1] = np.random.choice((-1,0,1),d-2) # effect of M on Y
    elif stype == 'interacted': # Erdos-Renyi model
        prob = float(degree) / (d - 1)
        B = np.tril((np.random.rand(d, d) < prob).astype(float), k=-1)
        
        # random permutation
        P = np.random.permutation(np.eye(d, d))  # permutes first axis only
        B_perm = P.T.dot(B).dot(P)
        U = np.random.uniform(low=w_range[0], high=w_range[1], size=[d, d])
        U[np.random.rand(d, d) < 0.5] *= -1
        W = (B_perm != 0).astype(float) * U
        
        # remove all in-edges (from precedent nodes) of the first node as A
        W[:, 0] = 0
        # remove all out-edges (from descendent nodes) of the last node as Y
        W[d-1, :] = 0
    else:
        raise ValueError("stype must be one of ('simple','parallel','interacted')")
    W = reset_mediators(W,0)
    W_X = add_X_to_graph(W,simple = (stype == "simple"), p=p, seed=xseed, noint=noint,
                         moderator=moderator, zero_chance=zero_chance)
    # fixed zero chance for now
    if XM: W_X = add_XM_to_graph(W_X,p=p,s=s,seed=xseed,zero_chance=0.8)
    G = nx.DiGraph(W_X)
    return G,W,W_X

def simulate_lsem(G: nx.DiGraph, n: int,p: int = 1, seed: int = 1, s: int = 1,
                  noise_scale: float = 0.5, baseline: float = 1.0,
                  noise=1) -> np.ndarray:
    """
    Simulate LSEM data given DAG.
        
    Args:
    G: DAG of interest
    n: number of samples
    p: number of potential moderators
    seed: seed for random functions
    noise_scale: variance
    baseline: baseline value for response
    noise: whether or not to add noise
        
    Returns:
    data: simulated LSEM data with dimension n x d x 1 where d is the number of nodes in G
    """
    np.random.seed(seed)
    W = nx.to_numpy_array(G)
    temp = W[:,:]
    temp[(p+1):(2*p+1),(p+1):(2*p+1)] = 0
    d = W.shape[0]
    X = np.zeros([n, d, 1])
    ordered_vertices = list(nx.topological_sort(nx.DiGraph(temp)))
    assert len(ordered_vertices) == d
    rank_A = ordered_vertices.index(p)

    for j in range(p):
        X[:,j,0] = noise*np.random.normal(scale=noise_scale, size=n)
    for j in ordered_vertices:
        if j < p:
            continue
        elif j == p:
            X[:, j, 0] = X[:, :, 0].dot(W[:, j]) + noise*np.random.normal(scale=noise_scale, size=n)
            X[:, (p+1):(2*p+1), 0] += (X[:, :p, 0].T * X[:, p, 0]).T # interaction
        elif p < j <= 2*p:
            continue
        elif 2*p+s+1 < j < d-1:
            continue
        else:
            X[:, j, 0] = X[:, :, 0].dot(W[:, j]) + noise*np.random.normal(scale=noise_scale, size=n)
    X[:, d-1, 0] += baseline
    return X - np.mean(X, axis=0)

def simulate_sem(G: nx.DiGraph, n: int,p: int = 1, seed: int = 1, s: int = 1,
                  noise_scale: float = 0.5, baseline: float = 1.0,
                  noise=1, f=lambda x: x, XM=False) -> np.ndarray:
    """
    Simulate LSEM data given DAG.
        
    Args:
    G: DAG of interest
    n: number of samples
    p: number of potential moderators
    seed: seed for random functions
    noise_scale: variance
    baseline: baseline value for response
    noise: whether or not to add noise
        
    Returns:
    data: simulated LSEM data with dimension n x d x 1 where d is the number of nodes in G
    """
    np.random.seed(seed)
    W = nx.to_numpy_array(G)
    temp = W[:,:]
    temp[(p+1):(2*p+1),(p+1):(2*p+1)] = 0
    d = W.shape[0]
    X = np.zeros([n, d, 1])
    ordered_vertices = list(nx.topological_sort(nx.DiGraph(temp)))
    assert len(ordered_vertices) == d
    rank_A = ordered_vertices.index(p)

    for j in range(p):
        X[:,j,0] = noise*np.random.normal(scale=noise_scale, size=n)
    for j in ordered_vertices:
        if j < p:
            continue
        elif j == p: #action
            X[:, j, 0] = f(X[:, :, 0]).dot(W[:, j]) + noise*np.random.normal(scale=noise_scale, size=n)
            X[:, (p+1):(2*p+1), 0] += (X[:, :p, 0].T * X[:, p, 0]).T # interaction
        elif p < j <= 2*p:
            continue
        elif (2*p+s+1 < j < d-1) and XM:
            continue
        else:
            X[:, j, 0] = f(X[:, :, 0]).dot(W[:, j]) + noise*np.random.normal(scale=noise_scale, size=n)
    X[:, d-1, 0] += baseline
    return X - np.mean(X, axis=0)

def simulate_sem_specific(G: nx.DiGraph, n: int,p: int = 1, seed: int = 1, s: int = 1,
                  noise_scale: float = 0.5, baseline: float = 1.0,
                  noise=1, f=lambda x: x, XM=False) -> np.ndarray:
    """
    Simulate LSEM data given DAG.
        
    Args:
    G: DAG of interest
    n: number of samples
    p: number of potential moderators
    seed: seed for random functions
    noise_scale: variance
    baseline: baseline value for response
    noise: whether or not to add noise
        
    Returns:
    data: simulated LSEM data with dimension n x d x 1 where d is the number of nodes in G
    """
    np.random.seed(seed)
    W = nx.to_numpy_array(G)
    temp = W[:,:]
    temp[(p+1):(2*p+1),(p+1):(2*p+1)] = 0
    d = W.shape[0]
    X = np.zeros([n, d, 1])
    ordered_vertices = list(nx.topological_sort(nx.DiGraph(temp)))
    assert len(ordered_vertices) == d
    rank_A = ordered_vertices.index(p)
    
    for j in range(p):
        X[:,j,0] = noise*np.random.normal(scale=noise_scale, size=n)
        x = X[:,j,0]
    for j in ordered_vertices:
        if j < p:
            continue
        elif j == p: #action
            X[:, j, 0] = x*W[0, j] + f(X[:, :, 0]).dot(W[:, j]) + noise*np.random.normal(scale=noise_scale, size=n)
            X[:, (p+1):(2*p+1), 0] = (X[:, :p, 0].T * X[:, p, 0]).T # interaction
        elif p < j <= 2*p:
            continue
        elif (2*p+s+1 < j < d-1):# and XM:
            continue
        else:
            X[:, j, 0] = x*W[0, j] + f(X[:, :, 0]).dot(W[:, j]) + noise*np.random.normal(scale=noise_scale, size=n)
    X[:, d-1, 0] += baseline
    return X - np.mean(X, axis=0)