import numpy as np
import numpy.typing as npt
from utils import *
from plot_utils import draw_dendrogram

def ffind_hce_level(H : npt.NDArray, rn=0, nK=None, on=True, **kwargs):
  ''' 
  Finds the optimal hierarchical level using the HCE method from a linkage matrix.

  Parameters
  ----------

  H : npt.NDArray
      Linkage matrix in Scipy format.
  rn : int, optional
      Number of renormalization steps to find the optimal hierarchical level. Default is 0.
  nK : int, optional
      Specify partition level with nK communties. If None, the optimal number of communities is determined.
  on : bool, optional
      If True, draws the dendrogram. Default is True.
  kwargs : dict, optional
      Additional keyword arguments to pass to the draw_dendrogram function.

  Returns
  -------

  labels : (npt.NDArray)
      Community memberships at the optimal level
  K : int
      Number of communities at the optimal level
  '''

  if H.ndim != 2 or H.shape[1] != 4:
    raise ValueError("H must be a 2D array with shape (N-1, 4).")
  
  if nK is None:
  
    hce = HCE(H, (H.shape[0] + 1))
    K, _ = get_best_hce_level(hce)

    i = 0
    while i < rn:
      hce = rHCE(H, (H.shape[0] + 1), K)
      K, _ = get_best_hce_level(hce)
      i += 1

  else:
    K = nK

  if on:
    draw_dendrogram(H, K, (H.shape[0] + 1), **kwargs)

  labels = fast_cut_tree(H, n_clusters=K)

  return labels, K

def find_hce_level(distM : npt.NDArray, method="average", nK=None, rn=0, on=True, **kwargs):
  ''' 
  Finds the optimal hierarchical level using the HCE method from a distance matrix.

  Parameters
  ----------

  distM : npt.NDArray
      Distance matrix.
  method : str, optional
        Method to use in Scipy's linkage function. Default is "average".
  nK : int, optional
      Specify partition level with nK communties. If None, the optimal number of communities is determined.
  rn : int, optional
      Number of renormalization steps to find the optimal hierarchical level. Default is 0.
  on : bool, optional
      If True, draws the dendrogram. Default is True.
  kwargs : dict, optional
      Additional keyword arguments to pass to the draw_dendrogram function.

  Returns
  -------

  labels : (npt.NDArray)
      Community memberships at the optimal level
  K : int
      Number of communities at the optimal level
  '''
  from scipy.cluster.hierarchy import linkage
  from scipy.spatial.distance import squareform
  
  distM2 = distM.copy()

  if np.isnan(distM2).any() or np.isinf(distM2).any():
    print("WARNING: There are nan or infinite values. The algorithm will replace them by max(distM[distM < inf]) + 1e-3.")
    maxD = np.nanmax(distM2[distM2 < np.inf])
    distM2[np.isnan(distM2)] = maxD + 1e-3
    distM2[np.isinf(distM2)] = maxD + 1e-3
  
  if distM2.ndim == 1:
    H = linkage(distM2, method=method)
  elif distM2.ndim == 2:
    H = linkage(squareform(distM2), method=method)
  else:
    raise ValueError("distM must be a condensed 1D distance matrix array or a 2D distance Matrix.")
  
  if nK is None:
  
    Sh4 = HCE(H, (H.shape[0] + 1))
    K, _ = get_best_hce_level(Sh4, on=False)

    i = 0
    while i < rn:
      Sh4 = rHCE(H, (H.shape[0] + 1), K)
      K, _ = get_best_hce_level(Sh4, on=False)
      i += 1

  else:
    K = nK

  if on:
    draw_dendrogram(H, K, (H.shape[0] + 1), **kwargs)

  labels = fast_cut_tree(H, n_clusters=K)

  return labels, K

def HCE(H : npt.NDArray, N : int, use_tqdm=False):
    '''
    Computes the hierarchical clustering of a dendrogram represented by a linkage matrix.
    Parameters
    ----------
    H : npt.NDArray
        Hierarchical clustering linkage matrix.
    N : int
        Number of nodes in the hierarchy.
    use_tqdm : bool, optional
        If True, uses tqdm for progress bar.
    Returns
    -------
    hce : dict
        Dictionary with the HCE metric for each level of the hierarchy.
    '''

    if H.shape[0] != N - 1:
        raise ValueError(f"Expected {N-1} rows in H, got {H.shape[0]}.")
    if H.shape[1] != 4:
        raise ValueError(f"Expected 4 columns in H, got {H.shape[1]}.")
    
    from tqdm import tqdm
    
    # Initialize the tree structure
    # Each node will store its size
    # This is a dictionary where keys are node indices and values are dictionaries with size
    # This allows for dynamic size updates as we merge nodes
    # T[(i)] = {"size": size_of_node_i}
    # Size is initialized to 0 for all nodes for node removal of the HCE process

    T = {(i) : {"size" : np.array(0)} for i in np.arange(N)}

    hce = {}

    if use_tqdm:
       iterator = tqdm(np.arange(N-1))
    else:
       iterator = np.arange(N-1)

    for i in iterator:
        nx, ny, h = int(H[i, 0]), int(H[i, 1]), H[i, 2]
        
        T[(N + i)] = {"size" : T[(nx)]["size"] + T[(ny)]["size"] + 1}   # Update size of the new node

        del T[(nx)]
        del T[(ny)]

        s = 0 
        for key in T.keys():
            if T[key]["size"] == 0: continue
            s += -(T[key]["size"] / (i+1)) * np.log(T[key]["size"]/ (i+1))  # Compute entropy
        
        if i < N - 2:
          if h != H[i+1, 2]:
           hce[(N-(i+1))] = s * (i+1) / (N-1) # apply community size normalization
        else:
          hce[(N-(i+1))] = s * (i+1) / (N-1) # apply community size normalization

    return hce


def rHCE(H : npt.NDArray, N : int, rN : int, use_tqdm=False):
    '''
    Computes the renormalized hierarchical clustering of a dendrogram represented by a linkage matrix.
    Parameters
    ----------
    H : npt.NDArray
        Hierarchical clustering linkage matrix.
    N : int
        Number of nodes in the hierarchy.
    rN : int
        Number of communities at the renormalization level..
    use_tqdm : bool, optional
        If True, uses tqdm for progress bar.
    Returns
    -------
    rhce : dict
        Dictionary with the renormalized HCE metric for each level of the hierarchy.
    '''

    if H.shape[0] != N - 1:
        raise ValueError(f"Expected {N-1} rows in H, got {H.shape[0]}.")
    if H.shape[1] != 4:
        raise ValueError(f"Expected 4 columns in H, got {H.shape[1]}.")
    if rN < 1 or rN >= N:
        raise ValueError(f"rN must be in the range [1, {N-1}]. Got {rN}.")
    
    from tqdm import tqdm

    # Initialize the tree structure
    # Each node will store its size
    # This is a dictionary where keys are node indices and values are dictionaries with size
    # This allows for dynamic size updates as we merge nodes
    # T[(i)] = {"size": size_of_node_i}
    # Size is initialized to 0 for all nodes for node removal of the HCE process

    T = {(i) : {"size" : np.array(0)} for i in np.arange(N)}

    rhce = {}

    # Update the tree structure for the first N - rN nodes
    # These nodes will be removed in the renormalization process
    # We initialize their size to 0, as they will not contribute to the renormalization

    for i in np.arange(N - rN):
        nx, ny = int(H[i, 0]), int(H[i, 1])
        
        T[(N + i)] = {"size" : np.array(0)}

        del T[(nx)]
        del T[(ny)]

        rhce[(N-(i+1))] = 0

    if use_tqdm:
       iterator = tqdm(enumerate(np.arange(N - rN, N - 1)))
    else:
       iterator = enumerate(np.arange(N - rN, N - 1))

    for j, i in iterator:
        nx, ny, h = int(H[i, 0]), int(H[i, 1]), H[i, 2]
        
        T[(N + i)] = {"size" : T[(nx)]["size"] + T[(ny)]["size"] + 1}

        del T[(nx)]
        del T[(ny)]

        s = 0 
        for key in T.keys():
            if T[key]["size"] == 0: continue
            s += -(T[key]["size"] / (j+1)) * np.log(T[key]["size"]/ (j+1))  # Compute entropy
        
        if i < N - 2:
          if h != H[i+1, 2]:
            rhce[(N-(i+1))] = s * (j+1) / (rN-1)    # apply community size normalization
        else:
            rhce[(N-(i+1))] = s * (j+1) / (rN-1)    # apply community size normalization

    return rhce

def get_best_hce_level(hce : dict):
  '''
    Function that computes the number of communities and dissimilarity value with the highest
    HCE value from a HCE dictionary.

    Parameters
    ----------

    hce : dict
        Dictionary with the HCE metric for each level of the hierarchy.

    Returns
    -------
    max_K : int
        The number of communities with the highest HCE value.
    max_hce : float
        The highest HCE value.
  '''
  K = np.array(list(hce.keys())) # K is the number of communities per level
  k_sort = np.argsort(K)
  hce_values = np.array(list(hce.values()))
  K = K[k_sort]
  hce_values = hce_values[k_sort]

  argmax_hce = np.nanargmax(hce_values)

  max_K = K[argmax_hce]
  max_hce = hce_values[argmax_hce]

  return max_K, max_hce