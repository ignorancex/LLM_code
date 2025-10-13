import numpy as np
import numpy.typing as npt
from scipy.spatial.distance import squareform

def graph_cosine_similarity(u : npt.ArrayLike, v : npt.ArrayLike, i, j):
  '''
  Computes the cosine similarity between two vectors u and v, excluding the
  i-th and j-th elements respectively.

  Parameters
  ----------

  u : npt.ArrayLike
      First vector.
  v : npt.ArrayLike
      Second vector.
  i : int
      Index of the node i in the vector u.
  j : int
      Index of the node j in the vector v.

  Returns
  -------

  float
      Cosine similarity between u and v, excluding the specified elements.
  '''

  if u.ndim != 1 or v.ndim != 1:
    raise ValueError("Both u and v must be 1D arrays.")

  uc = u.copy()
  vc = v.copy()

  uc[j], vc[i] = 0, 0

  uj = u[j]
  vi = v[i]

  uv = np.dot(uc, vc) + uj*vi + u[i]*v[j]
  uu = np.dot(u, u)
  vv = np.dot(v, v)

  if uu == 0 or vv == 0: return -1
  else:
    return uv / np.sqrt(uu * vv)

def compute_dissimilarity_matrix(A : npt.NDArray, f=graph_cosine_similarity):
  ''' Computes the dissimilarity matrix for a given adjacency matrix A.
  Parameters
  ----------

  A : npt.NDArray
      Adjacency matrix of the graph.
  f : function, optional
      Function to compute the dissimilarity between two nodes. Default is
      graph_cosine_similarity.

  Returns
  -------

  npt.NDArray
      Dissimilarity matrix, a condensed 1D array of distances between nodes.
  '''

  if A.ndim != 2 or A.shape[0] != A.shape[1]:
    raise ValueError("Input must be a square adjacency matrix.")
  
  nodes = A.shape[0]
  DistA = np.zeros(nodes*(nodes-1)//2)
  e = 0
  for i in np.arange(nodes):
    for j in np.arange(i+1, nodes):
      DistA[e] = f(A[i], A[j], i, j)
      e += 1

  DistA = np.clip(DistA, -1, 1)
  DistA = np.sqrt(2 * (1 - DistA))
  DistA[np.isnan(DistA)] = 2
  return DistA

def fast_cut_tree(H : npt.NDArray, n_clusters=None, height=None):
  '''
  Similar to scipy.cluster.hierarchy function cut_tree, but optimized.
    Parameters
    ----------

    H : npt.NDArray
        Hierarchical clustering linkage matrix.
    n_clusters : int, optional
        Number of clusters to form. If None, height must be specified.
    height : float, optional
        Threshold to apply when forming clusters. If None, n_clusters must be specified.

    Returns
    -------

    partition : npt.NDArray
        Array of cluster labels for each node in the hierarchy.
  '''

  if H.ndim != 2 or H.shape[1] != 4:
    raise ValueError("H must be a linkage matrix with shape (n-1, 4).")

  if n_clusters is None and height is None:
    raise ValueError("n_clusters or height must be given.")
  elif n_clusters is not None and height is None:
    thrd = n_clusters
    thrd_t = 0
  elif height is not None and n_clusters is None:
    thrd = height
    thrd_t = 1
  else:
     pass
    
  N = H.shape[0]+1
  T = {(i) : [i] for i in np.arange(N)}

  K = N
  i = 0

  while True:
    if thrd_t == 0:
      if K <= thrd: break
    else:
      if H[i, 2] > thrd: break

    nx, ny = int(H[i, 0]), int(H[i, 1])

    T[(N+i)] = T[(nx)] + T[(ny)]
    
    del T[(nx)]
    del T[(ny)]
    
    i += 1
    K -= 1
  
  partition = np.zeros(N, dtype=np.int64)
  for key, val in T.items():
    members = np.array(val)
    partition[members] = key
  
  return partition

def linear_partition(partition : npt.ArrayLike):
  ''' 
  Filters a partition to renumber the communities linearly.
  Parameters
  ----------

  partition : npt.ArrayLike
      Array of community labels for each node.

  Returns
  -------

  npt.NDArray
      Filtered partition with singletons replaced by -1 and renumbered.'''
  
  par = partition.copy()
  new_partition = par
  ndc = np.unique(par)
  for i, c in enumerate(ndc):
    new_partition[par == c] = i
  return new_partition

def filter_partition(partition : npt.ArrayLike):
  ''' Filters a partition to remove singleton communities and renumber the communities.

  Parameters
  ----------

  partition : npt.ArrayLike
      Array of community labels for each node.

  Returns
  -------

  npt.NDArray
      Filtered partition with singletons replaced by -1 and renumbered.'''
  par = partition.copy()
  from collections import Counter
  fq = Counter(par)
  for i in fq.keys():
    if fq[i] == 1: par[par == i] = -1
  new_partition = par
  ndc = np.unique(par[par != -1])
  for i, c in enumerate(ndc):
    new_partition[par == c] = i
  return new_partition

class HNRG:
  '''Hierarchical Nested Random Graph (HNRG) model.'''
  def __init__(self, N : int, R : int, L : int, kav=16, rho=1, seed=None):
    ''' 
    Parameters
    ----------

    N : int
        Number of nodes in the lowest level of the hierarchy.
    R : int
        Branching factor of the hierarchy.
    L : int
        Number of levels in the hierarchy.
    kav : int, optional
        Average degree of the network. Default is 16.
    rho : float, optional
        (cohesivness) Ratio of the sum  of expected connections of a node
        in coarse-grained communities from a partition level and the
        expected number of connections at a community at the level of the
        hierarchy. Default is 1.
    seed : int, optional
        Random seed for reproducibility. Default is None.
    '''
    self.N = N
    self.R = R
    self.L = L

    if seed is not None:
      np.random.seed(seed)

    self.N = N *((R+1) ** L)

    # Create community labels for the hierarchy

    self.hierarchical_communities_labels = np.zeros(self.N)
    for l in np.arange(L):
      nl = N * (R+1) ** (L - l - 1)
      communities = np.repeat(np.arange((R+1)**(l + 1)), nl)
      self.hierarchical_communities_labels = np.vstack([self.hierarchical_communities_labels, communities])
    
    self.hierarchical_communities_labels = self.hierarchical_communities_labels.T

    # Create Community size dictionary Sx

    self.Sx = [(R) * N * ((R+1) ** (L-1))] + [N * ((R+1) ** (L-i-1)) for i in np.arange(L)]
    self.Sx = {i : self.Sx[i] for i in range(len(self.Sx))}

    # Create the probability px

    self.px = {i : (rho ** (L-i) / (1 + rho) ** (L-i+1)) * (kav / (self.Sx[i]-1)) for i in np.arange(1, L+1)}
    self.px[0] = (rho ** (L) / (1 + rho) ** (L)) * (kav / (self.Sx[0]))

    if self.px[L] > 1:
       raise ValueError(f"p(L) = {self.px[L]} > 1. Choose a smaller value of kav or a larger value of rho.")

    condensed_N = int(self.N * (self.N - 1) // 2)
    pij = np.zeros(condensed_N)

    x = 0
    for i in np.arange(self.N):
      for j in np.arange(i+1, self.N):
        ci = self.hierarchical_communities_labels[i]
        cj = self.hierarchical_communities_labels[j]
        pij[x] = [u for u in np.arange(L+1) if ci[u] == cj[u]][-1]
        pij[x] = self.px[pij[x]]
        x += 1

    self.A = (pij > np.random.uniform(0, 1, size=condensed_N)).astype(int)
    self.A = squareform(self.A)