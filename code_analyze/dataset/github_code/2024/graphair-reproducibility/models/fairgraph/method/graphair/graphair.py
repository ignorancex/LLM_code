import os
import numpy as np
import scipy.sparse as sp
import torch
import wandb
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import GraphSAINTRandomWalkSampler
from torch_geometric.utils import (
    from_scipy_sparse_matrix,
    to_dense_adj,
    to_torch_sparse_tensor,
)

from .classifier import Classifier
from .aug_module import aug_module
from .GCN import GCN, GCN_Body

from models.fairgraph.utils.utils import (
    accuracy,
    fair_metric,
    scipysp_to_pytorchsp,
    spearman_correlation,
    weighted_homophily,
)

class graphair(nn.Module):
    """
    This class implements the Graphair model for node classification.
    
    Args:
        dataset (Dataset): Dataset object containing the dataset to be used.
        config (dict): Dictionary containing hyperparameters for the model.
        device (torch.device): Device to run the model on.
        logger (wandb.run): Wandb logger object to log metrics.
    """

    def __init__(
        self,
        dataset,
        config,
        device,
        logger,
    ) -> None:
        super(graphair, self).__init__()
        self.aug_model, self.f_encoder, self.sens_model, self.classifier = self.init_modules(dataset, config, device)
        self.dataset = dataset
        self.logger = logger
        self.config = config

        self.alpha = config.alpha
        self.beta = config.beta
        self.gamma = config.gamma
        self.lam = config.lam
        self.disable_ep = config.disable_ep
        self.disable_fm = config.disable_fm

        self.criterion_sens = nn.BCEWithLogitsLoss()
        self.criterion_cont = nn.CrossEntropyLoss()
        self.criterion_recons = nn.MSELoss()

        self.optimizer_s = torch.optim.Adam(
            self.sens_model.parameters(), lr=config.model_lr, weight_decay=1e-5
        )

        FG_params = [
            {"params": self.aug_model.parameters(), "lr": config.model_lr},
            {"params": self.f_encoder.parameters()},
        ]
        self.optimizer = torch.optim.Adam(FG_params, lr=config.model_lr, weight_decay=config.wd)

        self.optimizer_aug = torch.optim.Adam(
            self.aug_model.parameters(), lr=config.model_lr, weight_decay=config.wd
        )
        self.optimizer_enc = torch.optim.Adam(
            self.f_encoder.parameters(), lr=config.model_lr, weight_decay=config.wd
        )
        
        num_hidden=64
        num_proj_hidden=64
        self.fc1 = torch.nn.Linear(num_hidden, num_proj_hidden)
        self.fc2 = torch.nn.Linear(num_proj_hidden, num_hidden)

        self.optimizer_classifier = torch.optim.Adam(
            self.classifier.parameters(), lr=config.classifier_lr, weight_decay=config.wd
        )

    def init_modules(self, dataset, config, device):
        features = dataset.features
        
        aug_model = aug_module(features, n_hidden=64, temperature=1).to(device)
        f_encoder = GCN_Body(
            in_feats=features.shape[1],
            n_hidden=config.model_hidden,
            out_feats=64,
            dropout=0.1,
            nlayer=2,
        ).to(device)
        sens_model = GCN(
            in_feats=features.shape[1], n_hidden=64, out_feats=64, nclass=1
        ).to(device)
        classifier = Classifier(
            input_dim=64, hidden_dim=config.classifier_hidden
        )
        
        return aug_model, f_encoder, sens_model, classifier

    def projection(self, z):
        z = F.elu(self.fc1(z))
        return self.fc2(z)

    # this is the constrastive loss function!
    def info_nce_loss_2views(self, features):
        batch_size = int(features.shape[0] / 2)

        labels = torch.cat([torch.arange(batch_size) for i in range(2)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.cuda()

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).cuda()
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(
            similarity_matrix.shape[0], -1
        )

        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        negatives = similarity_matrix[~labels.bool()].view(
            similarity_matrix.shape[0], -1
        )

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        temperature = 0.07
        logits = logits / temperature
        return logits, labels

    def forward(self, adj, x):
        assert sp.issparse(adj)
        if not isinstance(adj, sp.coo_matrix):
            adj = sp.coo_matrix(adj)
        adj.setdiag(1)
        degrees = np.array(adj.sum(1))
        degree_mat_inv_sqrt = sp.diags(np.power(degrees, -0.5).flatten())
        adj_norm = degree_mat_inv_sqrt @ adj @ degree_mat_inv_sqrt
        adj_norm = scipysp_to_pytorchsp(adj_norm)

        adj = adj_norm.cuda()
        return self.f_encoder(adj, x)
    
    def fit(self):
        if self.dataset.batch:
            self.fit_batch(
                epochs = self.config.epochs,
                adj = self.dataset.adj,
                x = self.dataset.features,
                sens = self.dataset.sens,
                idx_sens = self.dataset.idx_sens_train,
                warmup=50,
                adv_epoches=1,
            )
        else:
            self.fit_whole(
                epochs = self.config.epochs,
                adj = self.dataset.adj,
                x = self.dataset.features,
                sens = self.dataset.sens,
                idx_sens = self.dataset.idx_sens_train,
                warmup=50,
                adv_epoches=1,
            )

    def fit_whole(self, epochs, adj, x, sens, idx_sens, warmup=None, adv_epoches=1):
        assert sp.issparse(adj)
        if not isinstance(adj, sp.coo_matrix):
            adj = sp.coo_matrix(adj)
        adj.setdiag(1)
        adj_orig = scipysp_to_pytorchsp(adj).to_dense()
        norm_w = adj_orig.shape[0] ** 2 / float(
            (adj_orig.shape[0] ** 2 - adj_orig.sum()) * 2
        )
        degrees = np.array(adj.sum(1))
        degree_mat_inv_sqrt = sp.diags(np.power(degrees, -0.5).flatten())
        adj_norm = degree_mat_inv_sqrt @ adj @ degree_mat_inv_sqrt
        adj_norm = scipysp_to_pytorchsp(adj_norm)

        adj = adj_norm.cuda()

        if warmup:
            for _ in range(warmup):
                adj_aug, x_aug, adj_logits = self.aug_model(
                    adj, x, adj_orig=adj_orig.cuda()
                )
                edge_loss = norm_w * F.binary_cross_entropy_with_logits(
                    adj_logits, adj_orig.cuda()
                )

                feat_loss = self.criterion_recons(x_aug, x)
                recons_loss = edge_loss + self.lam * feat_loss

                self.optimizer_aug.zero_grad()
                with torch.autograd.set_detect_anomaly(True):
                    recons_loss.backward(retain_graph=True)
                self.optimizer_aug.step()


        for epoch_counter in range(epochs):
            ### generate fair view
            adj_aug, x_aug, adj_logits = self.aug_model(
                adj, x, adj_orig=adj_orig.cuda()
            )

            if self.disable_fm:
                x_aug = x
            if self.disable_ep:
                adj_aug = adj

            ### extract node representations
            h = self.projection(self.f_encoder(adj, x))
            h_prime = self.projection(self.f_encoder(adj_aug, x_aug))

            ## update sens model
            adj_aug_nograd = adj_aug.detach()
            x_aug_nograd = x_aug.detach()

            self.adj_aug = adj_aug_nograd
            self.x_aug = x_aug_nograd
            self.sens = sens

            if epoch_counter == 0:
                sens_epoches = adv_epoches * 10
            else:
                sens_epoches = adv_epoches
            for _ in range(sens_epoches):
                s_pred, _ = self.sens_model(adj_aug_nograd, x_aug_nograd)
                senloss = self.criterion_sens(
                    s_pred[idx_sens], sens[idx_sens].unsqueeze(1).float()
                )
                self.optimizer_s.zero_grad()
                senloss.backward()
                self.optimizer_s.step()
            s_pred, _ = self.sens_model(adj_aug, x_aug)
            senloss = self.criterion_sens(
                s_pred[idx_sens], sens[idx_sens].unsqueeze(1).float()
            )

            ## update aug model
            logits, labels = self.info_nce_loss_2views(torch.cat((h, h_prime), dim=0))
            contrastive_loss = self.criterion_cont(logits, labels)

            ## update encoder
            edge_loss = norm_w * F.binary_cross_entropy_with_logits(
                adj_logits, adj_orig.cuda()
            )

            feat_loss = self.criterion_recons(x_aug, x)
            recons_loss = edge_loss + self.lam * feat_loss
            loss = (
                self.beta * contrastive_loss
                + self.gamma * recons_loss
                - self.alpha * senloss
            )
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.logger.log(
                {
                    "loss/sens_loss": senloss.item(),
                    "loss/contrastive_loss": contrastive_loss.item(),
                    "loss/edge_reconstruction_loss": edge_loss.item(),
                    "loss/feature_reconstruction_loss": feat_loss.item()
                }
            )

            print(
                "Epoch: {:04d}".format(epoch_counter + 1),
                "sens loss: {:.4f}".format(senloss.item()),
                "contrastive loss: {:.4f}".format(contrastive_loss.item()),
                "edge reconstruction loss: {:.4f}".format(edge_loss.item()),
                "feature reconstruction loss: {:.4f}".format(feat_loss.item()),
            )

        save_path = "./checkpoint/graphair_{}_alpha{}_beta{}_gamma{}_lambda{}".format(
            self.dataset, self.alpha, self.beta, self.gamma, self.lam
        )
        torch.save(self.state_dict(), save_path)

    def fit_batch(self, epochs, adj, x, sens, idx_sens, warmup=None, adv_epoches=1):
        assert sp.issparse(adj)
        norm_w = adj.shape[0] ** 2 / float((adj.shape[0] ** 2 - adj.sum()) * 2)

        idx_sens = idx_sens.cpu().numpy()
        sens_mask = np.zeros((x.shape[0], 1))
        sens_mask[idx_sens] = 1.0
        sens_mask = torch.from_numpy(sens_mask)

        edge_index, _ = from_scipy_sparse_matrix(adj)
        
        save_dir = "./checkpoint/{}".format(self.dataset)
        # check if save_dir exists, if not create it
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        miniBatchLoader = GraphSAINTRandomWalkSampler(
            Data(
                x=x,
                edge_index=edge_index,
                sens=sens,
                sens_mask=sens_mask,
                deg=torch.tensor(np.array(adj.sum(1)).flatten()),
            ),
            batch_size=1000,
            walk_length=3,
            sample_coverage=500,
            num_workers=0,
            save_dir=save_dir,
        )

        def normalize_adjacency(adj, deg):
            # Calculate the degrees
            row, col = adj.indices()
            edge_weight = (
                adj.values() if adj.values() is not None else torch.ones(row.size(0))
            )
            # degree = torch_scatter.scatter_add(
            #     edge_weight, row, dim=0, dim_size=adj.size(0)
            # )

            # Inverse square root of degree matrix
            degree_inv_sqrt = deg.pow(-0.5)
            degree_inv_sqrt[degree_inv_sqrt == float("inf")] = 0

            # Normalize
            row_inv = degree_inv_sqrt[row]
            col_inv = degree_inv_sqrt[col]
            norm_edge_weight = edge_weight * row_inv * col_inv

            # Create the normalized sparse tensor
            adj_norm = torch.sparse.FloatTensor(
                torch.stack([row, col]), norm_edge_weight, adj.size()
            )
            return adj_norm

        if warmup:
            for _ in range(warmup):
                for data in miniBatchLoader:
                    data = data.cuda()
                    sub_adj = normalize_adjacency(
                        to_torch_sparse_tensor(data.edge_index, data.edge_norm),
                        data.deg.float(),
                    ).cuda()
                    sub_adj_dense = to_dense_adj(
                        edge_index=data.edge_index, max_num_nodes=data.x.shape[0]
                    )[0].float()
                    adj_aug, x_aug, adj_logits = self.aug_model(
                        sub_adj, data.x, adj_orig=sub_adj_dense
                    )

                    edge_loss = norm_w * F.binary_cross_entropy_with_logits(
                        adj_logits, sub_adj_dense
                    )

                    feat_loss = self.criterion_recons(x_aug, data.x)
                    recons_loss = edge_loss + self.lam * feat_loss

                    self.optimizer_aug.zero_grad()
                    with torch.autograd.set_detect_anomaly(True):
                        recons_loss.backward(retain_graph=True)
                    self.optimizer_aug.step()

                    print(
                        "edge reconstruction loss: {:.4f}".format(edge_loss.item()),
                        "feature reconstruction loss: {:.4f}".format(feat_loss.item()),
                    )

        for epoch_counter in range(epochs):
            for data in miniBatchLoader:
                data = data.cuda()

                sub_adj = normalize_adjacency(
                    to_torch_sparse_tensor(data.edge_index, data.edge_norm),
                    data.deg.float(),
                ).cuda()
                sub_adj_dense = to_dense_adj(
                    edge_index=data.edge_index, max_num_nodes=data.x.shape[0]
                )[0].float()

                adj_aug, x_aug, adj_logits = self.aug_model(
                    sub_adj, data.x, adj_orig=sub_adj_dense
                )

                if self.disable_fm:
                    x_aug = data.x
                if self.disable_ep:
                    adj_aug = sub_adj

                ### extract node representations
                h = self.projection(self.f_encoder(sub_adj, data.x))
                h_prime = self.projection(self.f_encoder(adj_aug, x_aug))

                ### update sens model
                adj_aug_nograd = adj_aug.detach()
                x_aug_nograd = x_aug.detach()

                self.adj_aug = adj_aug_nograd
                self.x_aug = x_aug_nograd
                self.sens = data.sens

                mask = (data.sens_mask == 1.0).squeeze()

                if epoch_counter == 0:
                    sens_epoches = adv_epoches * 10
                else:
                    sens_epoches = adv_epoches
                for _ in range(sens_epoches):
                    s_pred, _ = self.sens_model(adj_aug_nograd, x_aug_nograd)
                    senloss = torch.nn.BCEWithLogitsLoss(
                        weight=data.node_norm, reduction="sum"
                    )(s_pred[mask].squeeze(), data.sens[mask].float())

                    self.optimizer_s.zero_grad()
                    senloss.backward()
                    self.optimizer_s.step()

                s_pred, _ = self.sens_model(adj_aug, x_aug)
                senloss = torch.nn.BCEWithLogitsLoss(
                    weight=data.node_norm, reduction="sum"
                )(s_pred[mask].squeeze(), data.sens[mask].float())

                ## update aug model
                logits, labels = self.info_nce_loss_2views(
                    torch.cat((h, h_prime), dim=0)
                )
                contrastive_loss = (
                    nn.CrossEntropyLoss(reduction="none")(logits, labels)
                    * data.node_norm.repeat(2)
                ).sum()

                ## update encoder
                edge_loss = norm_w * F.binary_cross_entropy_with_logits(
                    adj_logits, sub_adj_dense
                )

                feat_loss = self.criterion_recons(x_aug, data.x)
                recons_loss = edge_loss + self.lam * feat_loss
                loss = (
                    self.beta * contrastive_loss
                    + self.gamma * recons_loss
                    - self.alpha * senloss
                )
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            self.logger.log(
                {
                    "epoch": epoch_counter + 1,
                    "loss/sens_loss": senloss.item(),
                    "loss/contrastive_loss": contrastive_loss.item(),
                    "loss/edge_reconstruction_loss": edge_loss.item(),
                    "loss/feature_reconstruction_loss": feat_loss.item(),
                }
            )
            print(
                "Epoch: {:04d}".format(epoch_counter + 1),
                "sens loss: {:.4f}".format(senloss.item()),
                "contrastive loss: {:.4f}".format(contrastive_loss.item()),
                "edge reconstruction loss: {:.4f}".format(edge_loss.item()),
                "feature reconstruction loss: {:.4f}".format(feat_loss.item()),
            )

        save_path = "./checkpoint/graphair_{}".format(self.dataset)
        torch.save(self.state_dict(), save_path)

    def test(self):
        features = self.dataset.features
        adj = self.dataset.adj
        labels = self.dataset.labels
        epochs = self.config.test_epochs
        idx_train = self.dataset.idx_train
        idx_val = self.dataset.idx_val
        idx_test = self.dataset.idx_test
        sens = self.dataset.sens

        h = self.forward(adj, features)
        h = h.detach()
        acc_list = []
        dp_list = []
        eo_list = []
        # set custom metric for all classifier runs

        for i in range(5):
            torch.manual_seed(i * 10)
            np.random.seed(i * 10)

            # train classifier
            self.classifier.reset_parameters()
            best_acc = 0.0
            best_test = 0.0
            for epoch in range(epochs):
                self.classifier.train()
                self.optimizer_classifier.zero_grad()
                output = self.classifier(h)
                loss_train = F.binary_cross_entropy_with_logits(
                    output[idx_train], labels[idx_train].unsqueeze(1).float()
                )

                loss_train.backward()
                self.optimizer_classifier.step()

                self.classifier.eval()
                output = self.classifier(h)
                acc_val = accuracy(output[idx_val], labels[idx_val])
                acc_test = accuracy(output[idx_test], labels[idx_test])

                parity_val, equality_val = fair_metric(output, idx_val, labels, sens)
                parity_test, equality_test = fair_metric(output, idx_test, labels, sens)
                if epoch % 10 == 0:
                    self.logger.log(
                        {
                            f"classifier_run_{i}/": {
                                "acc_test": acc_test.item(),
                                "acc_val": acc_val.item(),
                                "dp_val": parity_val,
                                "dp_test": parity_test,
                                "eo_val": equality_val,
                                "eo_test": equality_test,
                            }
                        }
                    )
                print(
                        "Epoch [{}] Test set results:".format(epoch),
                        "acc_test= {:.4f}".format(acc_test.item()),
                        "acc_val: {:.4f}".format(acc_val.item()),
                        "dp_val: {:.4f}".format(parity_val),
                        "dp_test: {:.4f}".format(parity_test),
                        "eo_val: {:.4f}".format(equality_val),
                        "eo_test: {:.4f}".format(equality_test),
                    )
                if acc_val > best_acc:
                    best_acc = acc_val
                    best_test = acc_test
                    best_dp = parity_val
                    best_dp_test = parity_test
                    best_eo = equality_val
                    best_eo_test = equality_test

            print("Optimization Finished!")
            print(
                "Test results:",
                "acc_test= {:.4f}".format(best_test.item()),
                "acc_val: {:.4f}".format(best_acc.item()),
                "dp_val: {:.4f}".format(best_dp),
                "dp_test: {:.4f}".format(best_dp_test),
                "eo_val: {:.4f}".format(best_eo),
                "eo_test: {:.4f}".format(best_eo_test),
            )

            self.logger.log(
                {
                    'best/accuracy':best_test.item(),
                    'best/dp':best_dp_test,
                    'best/eo':best_eo_test
                }
            )
            
            acc_list.append(best_test.item())
            dp_list.append(best_dp_test)
            eo_list.append(best_eo_test)

        table = wandb.Table(
            columns=["accuracy", "accuracy_std", "dp", "dp_std", "eo", "eo_std"]
        )

        results = [
            np.mean(acc_list),
            np.std(acc_list),
            np.mean(dp_list),
            np.std(dp_list),
            np.mean(eo_list),
            np.std(eo_list),
        ]
        table.add_data(*results)
        self.logger.log({"table/results": table})

        self.logger.log(
            {
                "results/accuracy": np.mean(acc_list),
                "results/dp": np.mean(dp_list),
                "results/eo": np.mean(eo_list),
            }
        )

        print(
            "Avg results:",
            "acc: {:.4f} std: {:.4f}".format(np.mean(acc_list), np.std(acc_list)),
            "dp: {:.4f} std: {:.4f}".format(np.mean(dp_list), np.std(dp_list)),
            "eo: {:.4f} std: {:.4f}".format(np.mean(eo_list), np.std(eo_list)),
        )

        homophily_values = weighted_homophily(self.adj_aug.to_dense(), self.sens)
        spearman_correlations = spearman_correlation(self.x_aug, self.sens)

        return {
            "accuracy": [np.mean(acc_list), np.std(acc_list)],
            "dp": [np.mean(dp_list), np.std(dp_list)],
            "eo": [np.mean(eo_list), np.std(eo_list)],
            "homophily": [homophily_values],
            "spearman": [spearman_correlations],
        }
