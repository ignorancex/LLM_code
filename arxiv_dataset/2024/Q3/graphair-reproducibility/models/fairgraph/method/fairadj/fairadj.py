# @Author  : Peizhao Li
# @Contact : peizhaoli05@gmail.com

import os.path as osp
import gc
import scipy.sparse as sp
import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.utils import train_test_split_edges
from torch_geometric.loader import GraphSAINTRandomWalkSampler
from sklearn.metrics import accuracy_score, roc_auc_score

from .utils import preprocess_graph, project, find_link
from .optimizer import loss_function
from .gae import GCNModelVAE
from models.fairgraph.method.fairdrop.utils import prediction_fairness

import wandb


class fairadj:
    def __init__(
        self,
        dataset,
        config,
        device,
        logger,
    ):
        self.device = device
        self.dataset_name = dataset.name
        self.hidden1 = config.hidden1
        self.hidden2 = config.hidden2
        self.dropout = config.dropout
        self.lr = config.model_lr
        self.config = config
        self.logger = logger

        self.reset_attributes()

    def initialize_attributes(self):
        adj, self.data = self.load_dataset(self.dataset_name)
        self.features = torch.FloatTensor(self.data.x).cuda()
        self.labels = torch.LongTensor(self.data.y).cuda()
        self.n_nodes, self.feat_dim = self.features.shape

        # preprocess adjacency matrix, label matrix and find intra and inter group links
        self.adj_norm = preprocess_graph(adj).to(self.device)
        adj = adj + sp.eye(adj.shape[0])
        adj = sp.coo_matrix(adj)
        self.adj_label = torch.FloatTensor(adj.toarray()).to(self.device)
        self.intra_pos, self.inter_pos, self.intra_link_pos, self.inter_link_pos = (
            find_link(adj, self.labels.clone().cpu().detach().numpy())
        )

        # calculate positive weight and norm
        self.pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
        self.pos_weight = torch.Tensor([self.pos_weight]).to(self.device)
        self.norm = (
            adj.shape[0]
            * adj.shape[0]
            / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)
        )
        
        del adj
        
    def reset_attributes(self):
        self.data = None
        self.labels = None
        self.features = None
        self.adj_norm = None
        self.pos_weight = None
        self.norm = None

        self.intra_pos = None
        self.inter_pos = None
        self.intra_link_pos = None
        self.inter_link_pos = None
        self.n_nodes = None
        self.feat_dim = None

    def load_dataset(self, dataset_name):
        path = osp.join(
            osp.dirname(osp.realpath("__file__")), "..", "data", dataset_name
        )

        data = Planetoid(path, dataset_name, transform=T.NormalizeFeatures())[0]

        data = train_test_split_edges(data, val_ratio=0.2, test_ratio=0.3)
        labels = torch.LongTensor(data.y)
        edges = np.array(
            [
                data.train_pos_edge_index[0].numpy(),
                data.train_pos_edge_index[1].numpy(),
            ],
            dtype=np.int32,
        ).T
        adj = sp.coo_matrix(
            (np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
            shape=(labels.shape[0], labels.shape[0]),
            dtype=np.float32,
        )
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        
        del edges
        return adj, data

    # note: unused function
    def split_data(self):
        # temporary function for generating
        split = T.RandomLinkSplit(
            num_val=0,
            num_test=0.1,
            add_negative_train_samples=True,
            neg_sampling_ratio=0.4,
            split_labels=True,
        )

        train_data, val_data, test_data = split(self.data_orig)
        print("printing batches")

        clean = T.RemoveIsolatedNodes()
        train_data, test_data = clean(train_data), clean(test_data)

        dataloader = GraphSAINTRandomWalkSampler(
            data=self.train_data,
            batch_size=1000,
            walk_length=3,
            sample_coverage=500,
            num_workers=0,
        )
        return train_data, test_data, dataloader

    def fit(self):
        acc_auc = []
        fairness = []
        runs = 5
        for idx in range(runs):
            # reset memory
            torch.cuda.empty_cache()
            gc.collect()
            self.reset_attributes()
            print(f"Running on seed  {idx + 1} out of {runs}")
            
            # set seed
            seed = 10 * idx
            np.random.seed(seed)
            torch.manual_seed(seed)

            # instantiate attributes
            print('Initializing attributes...')
            self.initialize_attributes()

            # initialization
            print('Initializing model...')
            self.model = GCNModelVAE(
                self.feat_dim, self.hidden1, self.hidden2, self.dropout
            ).to(self.device)
            
            # gradient clipping
            for p in self.model.parameters():
                p.register_hook(lambda grad: torch.clamp(grad, -1, 1))
                
            # setup model and optimizer
            print('Setting up model and optimizer...')
            self.optimizer = optim.Adam(self.model.get_parameters(), lr=self.lr)
            self.model.train()

            print('Starting training!')
            for outer_epoch in range(self.config.outer_epochs):
                print("Starting with outer epoch: {:d} of {:d}".format((outer_epoch + 1), self.config.outer_epochs))
                for epoch in range(self.config.T1):
                    self.optimizer.zero_grad()
                    recovered, _, mu, logvar = self.model(self.features, self.adj_norm)
                    loss = loss_function(
                        preds=recovered,
                        labels=self.adj_label,
                        mu=mu,
                        logvar=logvar,
                        n_nodes=self.n_nodes,
                        norm=self.norm,
                        pos_weight=self.pos_weight,
                    )

                    loss.backward()
                    self.optimizer.step()

                    self.logger.log({f"run_{idx}/T1_loss": loss.item()})

                    print(
                        "Epoch in T1: [{:d}/{:d}];".format((epoch + 1), self.config.T1),
                        "Loss: {:.3f};".format(loss.item()),
                    )

                if self.dataset_name.lower() == "pubmed":
                    del loss, recovered, mu, logvar
                    torch.cuda.empty_cache()

                for epoch in range(self.config.T2):
                    self.adj_norm = self.adj_norm.requires_grad_(True)
                    recovered = self.model(self.features, self.adj_norm)[0]

                    if self.dataset_name.lower() == "pubmed":
                        self.intra_pos = None
                        self.inter_pos = None
                        intra_score = recovered[
                            self.intra_link_pos[:, 0], self.intra_link_pos[:, 1]
                        ].mean()
                        inter_score = recovered[
                            self.inter_link_pos[:, 0], self.inter_link_pos[:, 1]
                        ].mean()
                    else:
                        intra_score = recovered[
                            self.intra_pos[:, 0], self.intra_pos[:, 1]
                        ].mean()
                        inter_score = recovered[
                            self.inter_pos[:, 0], self.inter_pos[:, 1]
                        ].mean()

                    loss = F.mse_loss(intra_score, inter_score)

                    loss.backward()

                    print(
                        "Epoch in T2: [{:d}/{:d}];".format(epoch + 1, self.config.T2),
                        "Loss: {:.5f};".format(loss.item()),
                    )

                    self.logger.log({f"run_{idx}/T2_loss": loss.item()})

                    self.adj_norm = self.adj_norm.add(
                        self.adj_norm.grad.mul(-self.config.eta)
                    ).detach()

                    self.adj_norm = self.adj_norm.to_dense()
                    self.adj_norm = project(self.adj_norm)
                    self.adj_norm = self.adj_norm.to_sparse()

                    if self.dataset_name.lower() == "pubmed":
                        del recovered, loss, intra_score, inter_score
                        torch.cuda.empty_cache()
                        gc.collect()

            with torch.no_grad():
                acc_auc, fairness = self.evaluate(acc_auc, fairness)
            self.logger.log(
                {
                f"run_{idx}/accuracy": acc_auc[-1][0],
                f"run_{idx}/roc": acc_auc[-1][1],
                f"run_{idx}/dp_mixed": fairness[-1][0],
                f"run_{idx}/eo_mixed": fairness[-1][1],
                f"run_{idx}/dp_group": fairness[-1][2],
                f"run_{idx}/eo_group": fairness[-1][3],
                f"run_{idx}/dp_sub": fairness[-1][4],
                f"run_{idx}/eo_sub": fairness[-1][5],
             }
            )

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


    def evaluate(self, acc_auc, fairness):
        # using val split as test split lacks neg edge index
        test_edges_true = self.data.val_pos_edge_index
        test_edges_false = self.data.val_neg_edge_index
        edge_index_test = torch.cat([test_edges_true, test_edges_false], dim=-1)

        self.model.eval()

        with torch.no_grad():
            z = self.model(self.features, self.adj_norm)[1]

        emb = z.data.cpu().numpy()

        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        adj_rec = np.array(np.dot(emb, emb.T), dtype=np.float128)

        preds_pos = []

        for e in test_edges_true.T:
            preds_pos.append(sigmoid(adj_rec[e[0], e[1]]))

        preds_neg = []
        for e in test_edges_false.T:
            preds_neg.append(sigmoid(adj_rec[e[0], e[1]]))

        preds_all = np.hstack([preds_pos, preds_neg])
        labels_all = np.hstack([np.ones(len(preds_pos)), np.zeros(len(preds_neg))])

        auc = roc_auc_score(labels_all, preds_all)

        cut = [0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75]
        best_acc = 0
        best_cut = 0.5
        for i in cut:
            acc = accuracy_score(labels_all, preds_all >= i)
            if acc > best_acc:
                best_acc = acc
                best_cut = i
        f = prediction_fairness(
            edge_index_test, labels_all, preds_all >= best_cut, self.labels.cpu()
        )
        acc_auc.append([best_acc * 100, auc * 100])
        fairness.append([x * 100 for x in f])

        return acc_auc, fairness
