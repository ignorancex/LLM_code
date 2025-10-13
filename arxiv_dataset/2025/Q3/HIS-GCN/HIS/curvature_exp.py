from HIS.HISsampler import *
import networkx as nx
from GraphRicciCurvature.OllivierRicci import OllivierRicci
import scipy.sparse as sp


def convert_to_undirected(csr_adj):
    csr_adj = csr_adj.maximum(csr_adj.T)
    csr_adj.setdiag(0)
    csr_adj.eliminate_zeros()
    csr_adj.data.fill(1)
    return csr_adj

def read_npz_format(filename):
    return sp.load_npz(f'./datasets/{filename}/adj_full.npz').astype(bool)


def read_txt_format(filename):
    G = nx.Graph()
    with open(f"./HIS/data/{filename}.txt", 'r') as f:
        for line in f:
            node1, node2 = map(int, line.strip().split())
            G.add_edge(node1, node2)
    return nx.to_scipy_sparse_array(G)


if __name__ == '__main__':
    filename = "facebook_combined"    # Modify this parameter to test the .txt file of graph you want to test

    G_adj = read_txt_format(filename)

    # If you want to read the graph data in .npz format,
    # please replace the above code with the following code:
    # G_adj = read_npz_format(filename)

    G_adj = convert_to_undirected(G_adj)

    num_nodes = G_adj.shape[0]
    num_edges = G_adj.count_nonzero() // 2
    print(f"num_nodes: {num_nodes}")
    print(f"num_edges: {num_edges}")

    G = nx.from_scipy_sparse_array(G_adj)

    print(f"process {filename} ...")
    orc = OllivierRicci(G, alpha=0., verbose="DEBUG")
    edge_curvatures = orc.compute_ricci_curvature_edges(list(G.edges()))
    edge_curvature_values = list(edge_curvatures.values())
    average_edge_curvature = sum(edge_curvature_values) / len(edge_curvature_values) if edge_curvature_values else 0
    print(f"OllivierRicci curvature: {average_edge_curvature}")

    sampler = FFSampler(G_adj, None, 0.1, core_rate=1)
    sampler.evaluate_network()

    sub_graph_datas = sampler.sample_process(1000)
    per_nodes = sampler.per_nodes

    subgraphs = []
    sub_oliv_list = []
    for subgraph_data in sub_graph_datas:
        node_subgraph, _ = subgraph_data
        sub_G = G.subgraph(node_subgraph)

        sub_orc = OllivierRicci(sub_G, alpha=0., verbose="DEBUG")
        edge_curvatures = sub_orc.compute_ricci_curvature_edges(list(sub_G.edges()))
        edge_curvature_values = list(edge_curvatures.values())
        sub_oliv_list.extend(edge_curvature_values)
    print(f"subgraph OllivierRicci curvature：{sum(sub_oliv_list)/len(sub_oliv_list)}")

