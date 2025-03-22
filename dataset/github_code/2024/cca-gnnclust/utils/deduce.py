"""
This file re-uses implementation from https://github.com/yl-1993/learn-to-cluster
"""
import dgl
import numpy as np
import torch
from sklearn import mixture

from .density import density_to_peaks, density_to_peaks_vectorize

__all__ = [
    "peaks_to_labels",
    "edge_to_connected_graph",
    "decode",
    "build_next_level",
]


def _find_parent(parent, u):
    idx = []
    # parent is a fixed point
    while u != parent[u]:
        idx.append(u)
        u = parent[u]
    for i in idx:
        parent[i] = u
    return u


def edge_to_connected_graph(edges, num):
    parent = list(range(num))
    for u, v in edges:
        p_u = _find_parent(parent, u)
        p_v = _find_parent(parent, v)
        parent[p_u] = p_v

    for i in range(num):
        parent[i] = _find_parent(parent, i)
    remap = {}
    uf = np.unique(np.array(parent))
    for i, f in enumerate(uf):
        remap[f] = i
    cluster_id = np.array([remap[f] for f in parent])
    return cluster_id


def peaks_to_edges(peaks, dist2peak, tau):
    edges = []
    for src in peaks:
        dsts = peaks[src]
        dists = dist2peak[src]
        for dst, dist in zip(dsts, dists):
            if src == dst or dist >= 1 - tau:
                continue
            edges.append([src, dst])
    return edges


def peaks_to_labels(peaks, dist2peak, tau, inst_num):
    edges = peaks_to_edges(peaks, dist2peak, tau)
    pred_labels = edge_to_connected_graph(edges, inst_num)
    return pred_labels, edges


def get_edge_dist(g, threshold):
    if threshold == "prob":
        return g.edata["prob_conn"][:, 0]
    return g.edata["raw_affine"]


def tree_generation(ng):
    k = 1
    ng.ndata["keep_eid"] = torch.zeros((ng.num_nodes(), k), device=ng.device).long() - 1

    def message_func(edges):
        return {"mval": edges.data["edge_dist"], "meid": edges.data[dgl.EID]}

    def reduce_func(nodes):
        curr_k = min(nodes.mailbox["mval"].shape[1], k)
        ind = torch.topk(nodes.mailbox["mval"], k=curr_k, dim=1)[1]
        keep_eid = torch.full((nodes.batch_size(), k), fill_value=-1, device=ng.device)
        keep_eid[:, :curr_k] = nodes.mailbox["meid"].gather(1, ind)
        return {"keep_eid": keep_eid}

    # node_order = [nids.to(ng.device) for nids in dgl.traversal.topological_nodes_generator(ng, 0)]
    # ng.prop_nodes(node_order, message_func, reduce_func)
    rev_ng = dgl.reverse(ng, copy_edata=True)
    rev_ng.update_all(message_func, reduce_func)
    ng = dgl.reverse(rev_ng, copy_edata=True)

    eids = ng.ndata["keep_eid"]
    eids = eids[eids > -1]
    edges = ng.find_edges(eids)
    treeg = dgl.graph(edges, num_nodes=ng.num_nodes())
    return treeg


def peak_propogation(treeg):
    treeg.ndata["pred_labels"] = torch.zeros(treeg.num_nodes(), device=treeg.device).long() - 1
    peaks = torch.where(treeg.out_degrees() == 0)[0].cpu().numpy()
    treeg.ndata["pred_labels"][peaks] = torch.arange(peaks.shape[0], device=treeg.device)

    def message_func(edges):
        return {"mlb": edges.src["pred_labels"]}

    def reduce_func(nodes):
        return {"pred_labels": nodes.mailbox["mlb"][:, 0]}

    rev_ng = dgl.reverse(treeg, copy_edata=True)
    node_order = [nids.to(rev_ng.device) for nids in dgl.traversal.topological_nodes_generator(rev_ng, 0)]
    rev_ng.prop_nodes(node_order, message_func, reduce_func)
    treeg = dgl.reverse(rev_ng, copy_edata=True)
    pred_labels = treeg.ndata["pred_labels"].cpu().numpy()
    return peaks, pred_labels


def decode(
    g,
    tau,
    threshold,
    use_gt,
    ids=None,
    global_edges=None,
    global_num_nodes=None,
    global_peaks=None,
):
    # Edge filtering with tau and density
    den_key = "density" if use_gt else "pred_den"
    g = g.local_var()
    g.edata["edge_dist"] = get_edge_dist(g, threshold)
    g.apply_edges(
        lambda edges: {
            "keep": (edges.src[den_key] < edges.dst[den_key]).long()
            * (edges.data["edge_dist"] >= tau).long()
        }
    )
    eids = torch.where(g.edata["keep"] == 0)[0]
    ng = dgl.remove_edges(g, eids)

    # Tree generation
    ng.edata[dgl.EID] = torch.arange(ng.num_edges(), device=g.device)
    treeg = tree_generation(ng)
    # Label propogation
    peaks, pred_labels = peak_propogation(treeg)

    if ids is None:
        return pred_labels, peaks

    # Merge with previous layers
    src, dst = treeg.edges()
    new_global_edges = (
        global_edges[0] + ids[src.cpu().numpy()].tolist(),
        global_edges[1] + ids[dst.cpu().numpy()].tolist(),
    )
    global_treeg = dgl.graph(new_global_edges, num_nodes=global_num_nodes)
    global_peaks, global_pred_labels = peak_propogation(global_treeg)
    return (
        pred_labels,
        peaks,
        new_global_edges,
        global_pred_labels,
        global_peaks,
    )


def build_next_level(
    features, labels, peaks, cam_ids, global_features, global_pred_labels, global_peaks, global_cam_ids, global_xws, global_yws
):
    global_peak_to_label = global_pred_labels[global_peaks]
    global_label_to_peak = np.zeros_like(global_peak_to_label)
    for i, pl in enumerate(global_peak_to_label):
        global_label_to_peak[pl] = i
    cluster_ind = np.split(
        np.argsort(global_pred_labels),
        np.unique(np.sort(global_pred_labels), return_index=True)[1][1:],
    )
    cluster_features = np.zeros((len(peaks), global_features.shape[1]))
    cam_ids_oh = np.zeros((len(peaks), 4)).astype(np.int32)
    nxws = np.zeros((len(peaks), 1))
    nyws = np.zeros((len(peaks), 1))
    for pi in range(len(peaks)):
        cluster_features[global_label_to_peak[pi], :] = np.mean(
            global_features[cluster_ind[pi], :], axis=0
        )
        nxws[global_label_to_peak[pi], 0] = np.mean(global_xws[cluster_ind[pi]], axis=0)
        nyws[global_label_to_peak[pi], 0] = np.mean(global_yws[cluster_ind[pi]], axis=0)
        cam_ids_oh[global_label_to_peak[pi], :][np.unique(global_cam_ids[cluster_ind[pi]])] = 1
    features = features[peaks]
    labels = labels[peaks]
    cam_ids = cam_ids[peaks]
    return features, labels, cam_ids, cluster_features, nxws, nyws, cam_ids_oh
