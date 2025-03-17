import dgl
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F

from .focal_loss import FocalLoss
from .graphconv import GraphConv


class LANDER(nn.Module):
    def __init__(
        self,
        feature_dim,
        nhid,
        num_conv=4,
        dropout=0,
        use_GAT=True,
        K=1,
        use_cluster_feat=True,
        use_focal_loss=True,
        **kwargs
    ):
        super(LANDER, self).__init__()
        nhid_half = int(nhid / 2)
        classifier_dim = 8
        self.use_cluster_feat = use_cluster_feat
        self.use_focal_loss = use_focal_loss

        if self.use_cluster_feat:
            self.feature_dim = feature_dim * 2
        else:
            self.feature_dim = feature_dim

        input_dim = (feature_dim, nhid, nhid, nhid_half)
        output_dim = (nhid, nhid, nhid_half, nhid_half)
        self.conv = nn.ModuleList()
        self.conv.append(GraphConv(self.feature_dim, nhid, dropout, use_GAT, K))
        for i in range(1, num_conv):
            self.conv.append(
                GraphConv(input_dim[i], output_dim[i], dropout, use_GAT, K)
            )

        self.src_mlp = nn.Linear(output_dim[num_conv - 1], classifier_dim - 2)
        self.dst_mlp = nn.Linear(output_dim[num_conv - 1], classifier_dim - 2)

        self.classifier_conn = nn.Sequential(
            nn.PReLU(2 * classifier_dim),
            nn.Linear(2 * classifier_dim, classifier_dim),
            nn.PReLU(classifier_dim),
            nn.Linear(classifier_dim, 1)
        )

        if self.use_focal_loss:
            self.loss_conn = FocalLoss(2)
        else:
            self.loss_conn = nn.BCEWithLogitsLoss(reduction='none')
        self.loss_den = nn.MSELoss()

    def pred_conn(self, edges):
        src_feat = self.src_mlp(edges.src["conv_features"])
        dst_feat = self.dst_mlp(edges.dst["conv_features"])
        feat_cat = torch.cat((src_feat,
                            edges.src['xws'],
                            edges.src['yws'],
                            dst_feat,
                            edges.dst['xws'],
                            edges.dst['yws']), dim=1)

        pred_conn = self.classifier_conn(feat_cat)
        return {"pred_conn": pred_conn}

    def pred_den_msg(self, edges):
        prob = edges.data["prob_conn"][:, 0]
        res = edges.data["raw_affine"] * (prob - (1. - prob))
        return {"pred_den_msg": res}

    def forward(self, bipartite):
        if self.use_cluster_feat:
            neighbor_x = torch.cat(
                [
                    bipartite.ndata["features"],
                    bipartite.ndata["cluster_features"],
                ],
                axis=1,
            )
        else:
            neighbor_x = bipartite.ndata["features"]

        for i in range(len(self.conv)):
            neighbor_x = self.conv[i](bipartite, neighbor_x)

        bipartite.ndata["conv_features"] = neighbor_x

        bipartite.apply_edges(self.pred_conn)
        bipartite.edata["prob_conn"] = torch.sigmoid(bipartite.edata["pred_conn"])

        rev_bipartite = dgl.reverse(bipartite, copy_edata=True)
        rev_bipartite.update_all(
            self.pred_den_msg, fn.mean("pred_den_msg", "pred_den")
        )
        bipartite = dgl.reverse(rev_bipartite, copy_edata=True)

        return bipartite

    def compute_loss(self, bipartite):
        labels_conn = bipartite.edata["labels_conn"]
        loss_conn = self.loss_conn(
            bipartite.edata["pred_conn"], labels_conn.view(-1, 1).float()
        )

        return loss_conn
