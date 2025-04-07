import time
import torch
import torch.nn as nn

import gudhi


class BottleneckSimilarity(nn.Module):

    def __init__(self, _node_weights, _sent_weights):
        super().__init__()
        torch.manual_seed(17)
        self.src_emb = _node_weights
        self.trg_emb = _sent_weights

    def distance_matrix(self):
        # print('Computing distance matrices')
        src_dist = torch.sqrt(2 - 2 * torch.clamp(torch.mm(self.src_emb, torch.t(self.src_emb)), -1., 1.))
        trg_dist = torch.sqrt(2 - 2 * torch.clamp(torch.mm(self.trg_emb, torch.t(self.trg_emb)), -1., 1.))
        print('Done distance computation')
        return src_dist.cpu().detach().numpy(), trg_dist.cpu().detach().numpy()

    @staticmethod
    def compute_diagram(_x, homology_dim):
        print('Initializing Rips-Complex')
        rips_complex = gudhi.RipsComplex(_x)
        # print('Computing simplex tree')
        simplex_tree = rips_complex.create_simplex_tree(max_dimension=homology_dim)
        # print('Simplex tree computed')
        print('Computing persistence diagram')
        diag = simplex_tree.persistence()
        return [simplex_tree.persistence_intervals_in_dimension(w) for w in range(homology_dim)]

    def compute_distance(self):
        start_time = time.time()
        node_dist, sent_dist = self.distance_matrix()
        diag_node = self.compute_diagram(node_dist, homology_dim=1)
        diag_sent = self.compute_diagram(sent_dist, homology_dim=1)
        print("Filtration graph computed in: %.3f" % (time.time() - start_time))
        return min([gudhi.bottleneck_distance(x, y, e=0) for (x, y) in zip(diag_node, diag_sent)])
