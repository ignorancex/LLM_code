import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from utils import *

def draw_HCE(hce : dict, s=15, xshift=0, yshift=0, linewidth=1, fonttextsize=20, fontlabelsize=20, fontticklabelsize=12, color="r", f=None, xlabel=r"$K$", ylabel=r"HCE", ax=None):
  ''' Draws the HCE values for each level of the hierarchy.
  
  Parameters
  ----------
  hce : dict
      Dictionary with the HCE metric for each level of the hierarchy.
  s : int, optional
      Size of the marker for the maximum HCE value. Default is 15.
  xshift : float, optional
      Vertical shift for the text label of the maximum HCE value. Default is 0.
  yshift : float, optional
      Horizontal shift for the text label of the maximum HCE value. Default is 0.
  linewidth : float, optional
      Width of the line in the plot. Default is 1.
  color : str, optional
      Color of the marker for the maximum HCE value. Default is "r".
  f : callable, optional
      Function to apply to the x-axis values (K). If None, K is used as is.
  xlabel : str, optional
      Label for the x-axis. Default is r"$K$".
  ylabel : str, optional
      Label for the y-axis. Default is r"HCE".
  ax : matplotlib.axes.Axes, optional
      Axes object to draw the plot on. If None, uses the current axes.
      Default is None.
  '''

  K = np.array(list(hce.keys())) # K is the number of communities per level
  k_sort = np.argsort(K)
  hce_values = np.array(list(hce.values()))
  K = K[k_sort]
  hce_values = hce_values[k_sort]

  argmax_hce = np.nanargmax(hce_values)

  max_K = K[argmax_hce]
  max_hce = hce_values[argmax_hce]

  real_max_K = max_K

  if callable(f):
    K = f(K)
    f_max_K = f(max_K)
  else:
    f_max_K = max_K

  if ax is None:
    ax = plt.gca()

  ax.text(f_max_K + xshift, (max_hce + yshift), f"K = {real_max_K}", horizontalalignment='center', zorder=4, fontdict=dict(size=fonttextsize))
  ax.text(f_max_K + xshift, (max_hce + yshift), f"K = {real_max_K}", horizontalalignment='center', zorder=3, fontdict=dict(size=fonttextsize),  bbox=dict(edgecolor="k", facecolor='gray', alpha=0.1))
  ax.scatter([f_max_K], [max_hce], s=s, color=color, edgecolors='k', linewidth=linewidth, zorder=2)
  ax.plot(K, hce_values, zorder=1)

  ax.tick_params(axis='both', which='major', labelsize=fontticklabelsize)
  ax.set_ylabel(ylabel, fontdict=dict(size=fontlabelsize))
  ax.set_xlabel(xlabel, fontdict=dict(size=fontlabelsize))

  ax.minorticks_on()

  sns.despine(ax=ax)


def draw_dendrogram(Z : npt.NDArray, R : int, nodes, cmap_name="hls", leaf_font_size=20,  linewidth=0.5, remove_labels=False, ax=None, **kwargs):
      ''' Draws a dendrogram from the linkage matrix Z.
    
      Parameters
      ----------

      Z : npt.NDArray
          Linkage matrix in Scipy format.
      R : int
          Number of clusters to visualize.
      nodes : int
          Number of nodes in the dendrogram.
      cmap_name : str, optional
          Name of the colormap to use for coloring clusters. Default is "hls".
      leaf_font_size : int, optional
          Font size for leaf labels. Default is 20.
      linewidth : float, optional
          Width of the lines in the dendrogram. Default is 0.5.
      remove_labels : bool, optional
          If True, removes labels from the leaves of the dendrogram. Default is False.
      ax : matplotlib.axes.Axes, optional
          Axes object to draw the plot on. If None, uses the current axes.
          Default is None.
      **kwargs : dict, optional
          Additional keyword arguments to pass to the dendrogram function.
      '''

      print("Visualize Zdendrogram!!!")
      from scipy.cluster import hierarchy
      import matplotlib
      from matplotlib.colors import to_hex

      matplotlib.rcParams['lines.linewidth'] = linewidth
      if ax is None:
        ax = plt.gca()
      else:
        _, ax = plt.subplots(1, 1)

      if R > 1:
        partition = fast_cut_tree(Z, n_clusters=R).ravel()
        new_partition = filter_partition(partition)
        unique_clusters_id = np.sort(np.unique(new_partition))
        if -1 in unique_clusters_id:
          cm = sns.color_palette(cmap_name, len(unique_clusters_id)-1)
        else:
          cm = sns.color_palette(cmap_name, len(unique_clusters_id))
        dlf_col = "#808080"
        ##
        D_leaf_colors = {}
        for i, _ in enumerate(np.arange(nodes)):
          if new_partition[i] != -1:
            D_leaf_colors[i] = to_hex(cm[new_partition[i]])
          else: D_leaf_colors[i] = dlf_col
        ##
        link_cols = {}
        for i, i12 in enumerate(Z[:,:2].astype(int)):
          c1, c2 = (link_cols[x] if x > len(Z) else D_leaf_colors[x]
            for x in i12)
          link_cols[i+1+len(Z)] = c1 if c1 == c2 else dlf_col

        if not remove_labels:
          hierarchy.dendrogram(
            Z,
            labels=np.arange(nodes),
            color_threshold=Z[nodes - R, 2],
            link_color_func = lambda k: link_cols[k],
            leaf_rotation=90, leaf_font_size=leaf_font_size, ax=ax, **kwargs
          )
        else:
          hierarchy.dendrogram(
            Z,
            no_labels=True,
            color_threshold=Z[nodes - R, 2],
            link_color_func = lambda k: link_cols[k], leaf_rotation=90, ax=ax
          )
      else:
        deep_blue = to_hex(sns.color_palette("deep")[0])
        if not remove_labels:
          hierarchy.dendrogram(
            Z,
            labels=np.arange(nodes),
            color_threshold=np.inf,
            link_color_func = lambda k: deep_blue,
            leaf_rotation=90, leaf_font_size=leaf_font_size, ax=ax, **kwargs
          )
        else:
          hierarchy.dendrogram(
            Z,
            no_labels=True,
            color_threshold=np.inf,
            link_color_func = lambda k: deep_blue, leaf_rotation=90, ax=ax
          )
      plt.gca().set_ylabel("Distance")
      sns.despine()
      plt.xticks(rotation=90)

def plot_network_linkage(G : nx.Graph, H : npt.NDArray, nc : int, pos=None, edgewidth=0.5, node_communities=None, node_size=40, linewidth=2, ax=None, cm=None):
  ''' 
  Plots a network with nodes colored by their community membership.

  Parameters
  ----------

  G : nx.Graph
      NetworkX graph object representing the network.
  H : npt.NDArray
      Hierarchical clustering linkage matrix.
  nc : int
      Number of communities to visualize.
  pos : dict, optional
      Dictionary with node positions. If None, uses Kamada-Kawai layout.
      Default is None.
  node_communities : list or np.ndarray, optional
      List or array of node community labels. If None, computes communities from H.
      Default is None.
  node_size : int, optional
      Size of the nodes in the plot. Default is 40.
  linewidth : float, optional
      Width of the edges in the plot. Default is 2.
  ax : matplotlib.axes.Axes, optional
      Axes object to draw the plot on. If None, uses the current axes.
      Default is None.
  cm : dict, optional
      Dictionary mapping community labels to colors. If None, uses a color palette.
      Default is None.
  '''

  if ax is None:
    ax = plt.gca()

  from matplotlib.colors import to_hex
  if node_communities is None:
    labels = fast_cut_tree(H, n_clusters=nc)
    labels = filter_partition(labels)
  elif isinstance(node_communities, list) or isinstance(node_communities, np.ndarray):
    labels = node_communities
  else:
    raise ValueError("node_communities is not a list.")

  unique_labels = np.sort(np.unique(labels))
  dft_color = to_hex((0.5, 0.5, 0.5))

  if cm is None:
    if -1 in unique_labels:
      cm = list(sns.color_palette("hls", len(unique_labels)-1))
      cm = [dft_color] + cm
    else:
      cm = list(sns.color_palette("hls", len(unique_labels)))
      
    cm = {u: to_hex(c) for u, c in zip(unique_labels, cm)}

  if pos is None:
    pos = nx.kamada_kawai_layout(G)
  
  nx.draw_networkx_nodes(
    G, pos=pos, node_color=[cm[u] for u in labels],
    edgecolors="k", linewidths=edgewidth,
    node_size=node_size, ax=ax
  )

  nx.draw_networkx_edges(G, pos=pos, width=linewidth, ax=ax)
  ax.axis('off')

def tsne_layout(D : npt.NDArray, n_components=2, pf=0.4, random_state=None):
    '''
    Computes a t-SNE layout for a distance matrix D.

    Parameters
    ----------

    D : npt.NDArray
        Distance matrix.
    n_components : int, optional
        Number of dimensions for the t-SNE embedding. Default is 2.
    pf : float, optional
        Perplexity factor for t-SNE. Default is 30.
    random_state : int, optional
        Random seed for reproducibility. Default is None.

    Returns
    -------

    pos : dict
        Dictionary with node indices as keys and their corresponding
        t-SNE layout positions.
    '''

    from sklearn.manifold import TSNE

    N = D.shape[0]
    tsne = TSNE(n_components=n_components, perplexity=(N * pf), random_state=random_state)
    pos = tsne.fit_transform(D)
    pos = {i : pos[i] for i in np.arange(pos.shape[0])}

    return pos

def graph_coloring_palette(pos : dict, labels : npt.ArrayLike, alpha=0.75):
  '''
  Creates a graph-coloring palette for the communities in a graph.

  Parameters
  ----------

  pos : dict
      Dictionary with node positions.
  labels : npt.ArrayLike
      Array of community labels for each node.
  '''
  from scipy.spatial import Delaunay
  colors = {0 : plt.cm.Blues_r, 1 : plt.cm.Reds_r, 2 : plt.cm.Greens_r, 3 : plt.cm.Purples_r, 4: plt.cm.YlOrBr_r, 5 : plt.cm.RdPu_r}

  unique_communities = np.unique(labels)

  pos_array = np.array([p for p in pos.values()])

  G = nx.Graph()
  pos_graph = {}
  for u in unique_communities:
      pos_graph[u] = np.median(pos_array[labels == u], axis=0)
      G.add_node(u)

  coords = np.array([pos_graph[u] for u in G.nodes()])

  tri = Delaunay(coords)

  for t in tri.simplices:
      G.add_edge(t[0], t[1])
      G.add_edge(t[1], t[2])
      G.add_edge(t[2], t[0])

  is_planar, embedding_or_subgraph = nx.check_planarity(G)

  if not is_planar:
    print("The graph is not planar.")
    print("Kuratowski subgraph:", embedding_or_subgraph)
    raise ValueError("The graph is not planar. Cannot compute coloring palette.")
  
  coloring = nx.algorithms.coloring.greedy_color(G, strategy='largest_first')

  node_colors = {n : coloring[n] for n in G.nodes()}
  return {n : colors[node_colors[n]](alpha) for n in G.nodes()}