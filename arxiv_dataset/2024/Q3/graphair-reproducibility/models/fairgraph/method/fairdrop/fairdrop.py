import os.path as osp

import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from sklearn.metrics import accuracy_score, roc_auc_score
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GATConv, GCNConv
from torch_geometric.utils import negative_sampling, train_test_split_edges
import wandb

from .utils import (
    get_link_labels,
    prediction_fairness,
)


class GCN(torch.nn.Module):
    def __init__(self, in_channels, out_channels, hidden_dim):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_channels)

    def encode(self, x, pos_edge_index):
        x = F.relu(self.conv1(x, pos_edge_index))
        x = self.conv2(x, pos_edge_index)
        return x

    def decode(self, z, pos_edge_index, neg_edge_index):
        edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
        logits = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)
        return logits, edge_index


class GAT(torch.nn.Module):
    def __init__(self, in_channels, out_channels, hidden_dim, heads=1):
        super(GAT, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_dim, heads=heads)
        self.conv2 = GATConv(hidden_dim * heads, out_channels, heads=1)

    def encode(self, x, pos_edge_index):
        x = F.relu(self.conv1(x, pos_edge_index))
        x = self.conv2(x, pos_edge_index)
        return x

    def decode(self, z, pos_edge_index, neg_edge_index):
        edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
        logits = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)
        return logits, edge_index


class fairdrop:
    def __init__(
        self,
        dataset,
        config,
        device,
        logger,
    ):
        test_seeds = [20, 21, 22, 23, 24, 25]

        self.epochs = config.epochs
        self.lr = config.model_lr
        self.dataset_name = dataset.name
        self.gcn_hidden = config.hidden_gcn
        self.gat_hidden = config.hidden_gat
        self.delta = config.delta
        self.test_seeds = test_seeds
        self.device = device
        self.model_type = config.model_type
        self.gat_heads = config.heads
        self.logger = logger

        self.dataset = self.load_dataset()
        self.model = self.initialize_model()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def load_dataset(self):
        path = osp.join(
            osp.dirname(osp.realpath("__file__")), "..", "data", self.dataset_name
        )
        return Planetoid(path, self.dataset_name, transform=T.NormalizeFeatures())[0]

    def initialize_model(self):
        if self.model_type == "GCN":
            return GCN(self.dataset.num_features, 128, self.gcn_hidden).to(self.device)
        elif self.model_type == "GAT":
            return GAT(
                self.dataset.num_features, 128, self.gat_hidden, self.gat_heads
            ).to(self.device)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def fit(self):
        acc_auc = []
        fairness = []

        for random_seed in self.test_seeds:
            np.random.seed(random_seed)
            data = self.dataset.clone()
            print(data)
            protected_attribute = data.y
            print(protected_attribute)
            data.train_mask = data.val_mask = data.test_mask = data.y = None
            data = train_test_split_edges(data, val_ratio=0.1, test_ratio=0.2)
            data = data.to(self.device)

            num_classes = len(np.unique(protected_attribute))
            N = data.num_nodes

            Y = torch.LongTensor(protected_attribute).to(self.device)
            Y_aux = (
                Y[data.train_pos_edge_index[0, :]] != Y[data.train_pos_edge_index[1, :]]
            ).to(self.device)
            randomization = (
                torch.FloatTensor(self.epochs, Y_aux.size(0)).uniform_()
                < 0.5 + self.delta
            ).to(self.device)

            best_val_perf = test_perf = 0
            for epoch in range(1, self.epochs):
                # TRAINING
                neg_edges_tr = negative_sampling(
                    edge_index=data.train_pos_edge_index,
                    num_nodes=N,
                    num_neg_samples=data.train_pos_edge_index.size(1) // 2,
                ).to(self.device)

                if epoch == 1 or epoch % 10 == 0:
                    keep = torch.where(randomization[epoch], Y_aux, ~Y_aux)

                self.model.train()
                self.optimizer.zero_grad()

                z = self.model.encode(data.x, data.train_pos_edge_index[:, keep])
                link_logits, _ = self.model.decode(
                    z, data.train_pos_edge_index[:, keep], neg_edges_tr
                )
                tr_labels = get_link_labels(
                    data.train_pos_edge_index[:, keep], neg_edges_tr
                ).to(self.device)

                loss = F.binary_cross_entropy_with_logits(link_logits, tr_labels)

                self.logger.log({"loss/loss": loss})

                loss.backward()
                self.optimizer.step()

                # EVALUATION
                self.model.eval()
                perfs = []
                for prefix in ["val", "test"]:
                    pos_edge_index = data[f"{prefix}_pos_edge_index"]
                    neg_edge_index = data[f"{prefix}_neg_edge_index"]
                    with torch.no_grad():
                        z = self.model.encode(data.x, data.train_pos_edge_index)
                        link_logits, edge_idx = self.model.decode(
                            z, pos_edge_index, neg_edge_index
                        )
                    link_probs = link_logits.sigmoid()
                    link_labels = get_link_labels(pos_edge_index, neg_edge_index)
                    auc = roc_auc_score(link_labels.cpu(), link_probs.cpu())
                    perfs.append(auc)

                val_perf, tmp_test_perf = perfs
                if val_perf > best_val_perf:
                    best_val_perf = val_perf
                    test_perf = tmp_test_perf
                if epoch % 10 == 0:
                    log = "Epoch: {:03d}, Loss: {:.4f}, Val: {:.4f}, Test: {:.4f}"
                    print(log.format(epoch, loss, best_val_perf, test_perf))

            # FAIRNESS
            auc = test_perf
            cut = [0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75]
            best_acc = 0
            best_cut = 0.5
            for i in cut:
                acc = accuracy_score(link_labels.cpu(), link_probs.cpu() >= i)
                if acc > best_acc:
                    best_acc = acc
                    best_cut = i
            f = prediction_fairness(
                edge_idx.cpu(), link_labels.cpu(), link_probs.cpu() >= best_cut, Y.cpu()
            )
            acc_auc.append([best_acc * 100, auc * 100])
            fairness.append([x * 100 for x in f])

        ma = np.mean(np.asarray(acc_auc), axis=0)
        mf = np.mean(np.asarray(fairness), axis=0)

        sa = np.std(np.asarray(acc_auc), axis=0)
        sf = np.std(np.asarray(fairness), axis=0)

        table = wandb.Table(
            columns=[
                "accuracy",
                "accuracy_std",
                "roc",
                "roc_std",
                "dp_mixed",
                "dp_mixed_std",
                "eo_mixed",
                "eo_mixed_std",
                "dp_group",
                "dp_group_std",
                "eo_group",
                "eo_group_std",
                "dp_sub",
                "dp_sub_std",
                "eo_sub",
                "eo_sub_std",
            ],
        )
        results = [
            ma[0],
            sa[0],
            ma[1],
            sa[1],
            mf[0],
            sf[0],
            mf[1],
            sf[1],
            mf[2],
            sf[2],
            mf[3],
            sf[3],
            mf[4],
            sf[4],
            mf[5],
            sf[5],
        ]

        table.add_data(*results)

        self.logger.log({"table/metrics": table})

        self.logger.log(
            {
                "results/accuracy": ma[0],
                "results/roc": ma[1],
                "results/dp_mixed": mf[0],
                "results/eo_mixed": mf[1],
                "results/dp_group": mf[2],
                "results/eo_group": mf[3],
                "results/dp_sub": mf[4],
                "results/eo_sub": mf[5],
            }
        )

        print(f"ACC: {ma[0]:2f} +- {sa[0]:2f}")
        print(f"AUC: {ma[1]:2f} +- {sa[1]:2f}")

        print(f"DP mix: {mf[0]:2f} +- {sf[0]:2f}")
        print(f"EoP mix: {mf[1]:2f} +- {sf[1]:2f}")
        print(f"DP group: {mf[2]:2f} +- {sf[2]:2f}")
        print(f"EoP group: {mf[3]:2f} +- {sf[3]:2f}")
        print(f"DP sub: {mf[4]:2f} +- {sf[4]:2f}")
        print(f"EoP sub: {mf[5]:2f} +- {sf[5]:2f}")
