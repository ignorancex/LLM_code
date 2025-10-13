from HIS.HISsampler import *
import networkx as nx
import scipy.sparse as sp

def convert_to_undirected(csr_adj):
    csr_adj = csr_adj.maximum(csr_adj.T)
    csr_adj.setdiag(0)
    csr_adj.eliminate_zeros()
    csr_adj.data.fill(1)
    return csr_adj

def read_npz_format(filename):
    return sp.load_npz(f'./datasets/{filename}/adj_train.npz').astype(bool)


def read_txt_format(filename):
    G = nx.Graph()
    with open(f"./HIS/data/{filename}.txt", 'r') as f:
        for line in f:
            node1, node2 = map(int, line.strip().split())
            G.add_edge(node1, node2)
    return nx.to_scipy_sparse_array(G)


if __name__ == '__main__':
    filename = "ppi-large"  # Modify this parameter to test the dataset you want to test
    G_adj = read_npz_format(filename)

    # If you want to read the training graph data in .txt format,
    # please replace the above code with the following code:
    # G_adj = read_txt_format(filename)
    
    G_adj = convert_to_undirected(G_adj)
    degrees = np.array(G_adj.sum(1)).flatten()
    
    V_org = np.arange(G_adj.shape[0])
    max_edge_count = 0
    edge_count_dict = defaultdict(int)
    row, col = G_adj.nonzero()
    edges = list(zip(row, col))
    d_th = 0

    # You can modify edge_num to speed up the decomposition process. When there are enough edges in
    # the original graph. Modifying edge_rate will hardly affect the decomposition effect.
    edge_num = 20000

    edges_sampled = random.sample(edges,edge_num) if len(edges)> edge_num else edges
    t0 = time.time()
    print(f"process {filename} ...")
    for u, w in edges_sampled:
        u_degree = degrees[u]
        w_degree = degrees[w]
        for d in range(w_degree, u_degree):
            edge_count_dict[d] += 1
            if edge_count_dict[d] > max_edge_count:
                max_edge_count = edge_count_dict[d]
                d_th = d
    t1 = time.time()
    print(f"degree threshold: {d_th}")
    print(f"process time: {t1-t0:6.2f} sec")
