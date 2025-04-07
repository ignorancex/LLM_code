import torch
import torch.nn as nn

from itertools import chain
from tqdm import tqdm


class EigenvectorSimilarity(nn.Module):

    def __init__(self, _node_weights, _sent_weights):
        super().__init__()
        torch.manual_seed(17)
        self.src_emb = _node_weights
        self.trg_emb = _sent_weights

    def build_space_graphs(self):
        # TODO: this should really only happen on the top N most frequent "words"
        # TODO: what if we took random sub-samples and computed an average eigenvector similarity as a proxy, since
        # TODO: there isn't really a notion of frequency in this situation?
        # TODO: OR take top N words and top N hubs? given both spaces are represented as a graph, just look at
        # TODO: the nodes with the highest degree as they are informing of the critical structures of the space
        # TODO: interesting question: do we want to filter by the hubs or by the singletons?
        print('Building source space')
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        n_nodes = self.src_emb.shape[0]
        nodes_adj = torch.zeros((n_nodes, n_nodes))
        nodes_deg = torch.zeros((n_nodes, n_nodes))
        for j in tqdm(range(n_nodes)):
            cur_vec = self.src_emb[j, :].repeat(n_nodes, 1)
            outs = cos(cur_vec, self.src_emb)
            outs_cut = torch.where(outs > .90)[0]
            out_deg = len(outs_cut)
            nodes_deg[j][j] = out_deg
            for k in range(len(outs_cut)):
                cur_ind = outs_cut[k]
                nodes_adj[j][cur_ind] = 1
        self.node_adj = nodes_adj
        self.node_deg = nodes_deg

        print('Building target space')
        n_sents = self.trg_emb.shape[0]
        sents_adj = torch.zeros((n_sents, n_sents))
        sents_deg = torch.zeros((n_sents, n_sents))
        for j in tqdm(range(n_sents)):
            cur_vec = self.trg_emb[j, :].repeat(n_sents, 1)
            outs = cos(cur_vec, self.trg_emb)
            outs_cut = torch.where(outs > .90)[0]
            out_deg = len(outs_cut)
            sents_deg[j][j] = out_deg
            for k in range(len(outs_cut)):
                cur_ind = outs_cut[k]
                sents_adj[j][cur_ind] = 1
        self.sent_adj = sents_adj
        self.sent_deg = sents_deg

    def _compute_graph_laplacians(self):
        self.build_space_graphs()
        self.node_laplacian = self.node_deg - self.node_adj
        self.sent_laplacian = self.sent_deg - self.sent_adj

    @staticmethod
    def sum_of_squares(space_a, space_b):
        return torch.sum((space_a - space_b) ** 2)

    def compare_spaces(self):
        self._compute_graph_laplacians()
        sent_eigenvalues = torch.eig(self.sent_laplacian, eigenvectors=False)[0]
        sent_sorted_eigenvalues = torch.sort(sent_eigenvalues[:, 0], -1, True)[0]  # .numpy()
        sent_eignvalue_target = 0.9 * torch.sum(sent_sorted_eigenvalues).item()
        sent_cumsum = torch.cumsum(sent_sorted_eigenvalues, dim=0)
        sent_min_k = torch.min(torch.where(sent_cumsum > sent_eignvalue_target)[0]).item()
        #sev = list(chain.from_iterable(sent_eigenvalues.numpy()))
        #sent_min_k = 0
        #for k in range(len(sent_sorted_eigenvalues)):
        #    val = sent_sorted_eigenvalues[0:k].sum()
        #    check = [1 if val > x else 0 for x in sev]
        #    if sum(check) < .9 * len(sent_sorted_eigenvalues):
        #        sent_min_k = k
        print('Sent min k: {k}'.format(k=sent_min_k))

        node_eigenvalues = torch.eig(self.node_laplacian, eigenvectors=False)[0]
        node_sorted_eigenvalues = torch.sort(node_eigenvalues[:, 0], -1, True)[0]  # .numpy()
        node_eignvalue_target = 0.9 * torch.sum(node_sorted_eigenvalues).item()
        node_cumsum = torch.cumsum(node_sorted_eigenvalues, dim=0)
        node_min_k = torch.min(torch.where(node_cumsum > node_eignvalue_target)[0]).item()
        #nev = list(chain.from_iterable(node_eigenvalues.numpy()))
        #node_min_k = 0
        #for k in range(len(node_eigenvalues)):
        #    val = node_sorted_eigenvalues[0:k].sum()
        #    check = [1 if val > x else 0 for x in nev]
        #    if sum(check) < .9 * len(node_sorted_eigenvalues):
        #        node_min_k = k
        print('Node min k: {k}'.format(k=node_min_k))

        final_k = min(node_min_k, sent_min_k)
        node_k = node_sorted_eigenvalues[0: final_k]
        sent_k = sent_sorted_eigenvalues[0: final_k]
        delta = self.sum_of_squares(node_k, sent_k).item()
        return delta
