import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
import numpy as np
import wandb
from models.fairgraph.utils.utils import scipysp_to_pytorchsp,accuracy,fair_metric, auc
import random
from itertools import combinations_with_replacement

from torch_geometric.loader import GraphSAINTRandomWalkSampler
from torch_geometric.data import Data
from torch_geometric.utils import from_scipy_sparse_matrix, to_dense_adj, to_torch_sparse_tensor, to_edge_index, add_remaining_self_loops

from .classifier import Classifier
from .aug_module import aug_module
from .GCN import GCN, GCN_Body

class graphair(nn.Module):
    """
    This class implements the Graphair model for link prediction.
    
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
    ):
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
            {"params": self.aug_model.parameters(), "lr": 1e-4},
            {"params": self.f_encoder.parameters()},
        ]
        self.optimizer = torch.optim.Adam(
            FG_params, lr=config.model_lr, weight_decay=config.wd
        )

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
        aug_model = aug_module(features, n_hidden=config.model_hidden, temperature=1).to(device)
        f_encoder = GCN_Body(
            in_feats=features.shape[1],
            n_hidden=config.model_hidden,
            out_feats=64,
            dropout=0.1,
            nlayer=3,
        ).to(device)
        sens_model = GCN(
            in_feats=features.shape[1], n_hidden=config.model_hidden, out_feats=64, nclass=dataset.nclasses
        ).to(device)
        classifier = Classifier(
            input_dim=64, hidden_dim=config.classifier_hidden
        )
        return aug_model, f_encoder, sens_model, classifier
    
    def projection(self, z):
        z = F.elu(self.fc1(z))
        return self.fc2(z)
    
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
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
        
        temperature = 0.07
        logits = logits / temperature
        return logits, labels

    def forward(self, adj, x, sens):
        assert sp.issparse(adj)
        if not isinstance(adj, sp.coo_matrix):
            adj = sp.coo_matrix(adj)
        adj.setdiag(1)
        degrees = np.array(adj.sum(1))
        degree_mat_inv_sqrt = sp.diags(np.power(degrees, -0.5).flatten())
        adj_norm = degree_mat_inv_sqrt @ adj @ degree_mat_inv_sqrt
        adj_norm = scipysp_to_pytorchsp(adj_norm)
    
        adj = adj_norm.cuda()

        # Get node embeddings
        z = self.f_encoder(adj, x)
        
        # Create positive and negative link embeddings and labels
        pos_rows, pos_cols = adj.coalesce().indices()
        neg_rows, neg_cols = self.sample_negative_links(adj, num_neg_samples=pos_rows.shape[0])
        pos_labels = torch.ones(pos_rows.shape[0], dtype=torch.float32)
        neg_labels = torch.zeros(neg_rows.shape[0], dtype=torch.float32)

        rows = torch.cat([torch.tensor(pos_rows, device='cpu'), torch.tensor(neg_rows, device='cpu')], dim=0)
        cols = torch.cat([torch.tensor(pos_cols, device='cpu'), torch.tensor(neg_cols, device='cpu')], dim=0)

        
        link_embeddings = self.create_link_embeddings(z, rows, cols)
        labels = torch.cat([pos_labels, neg_labels], dim=0)

        # Create mixed dyadic groups
        groups_mixed = sens[rows] != sens[cols]

        # Create dyadic subgroups
        u = list(combinations_with_replacement(np.unique(sens.cpu()), r=2))
        groups_sub = []
        for i, j in zip(sens[rows], sens[cols]):
            for k, v in enumerate(u):
                if (i, j) == v or (j, i) == v:
                    groups_sub.append(k)
                    break
        groups_sub = np.asarray(groups_sub)

        return link_embeddings, labels, groups_mixed, groups_sub

    # ADDITIONAL FUNCTION
    def create_link_embeddings(self, node_embeddings, row_indices, col_indices):
        # Using a simple binary operator, e.g., element-wise addition
        link_embeddings = node_embeddings[row_indices] * node_embeddings[col_indices]
        return link_embeddings
    
    # ADDITIONAL FUNCTION
    def sample_negative_links(self, adj, num_neg_samples):
        n = adj.size(0)
        neg_rows, neg_cols = [], []
        while len(neg_rows) < num_neg_samples:
            i, j = random.randint(0, n - 1), random.randint(0, n - 1)
            
            # Check if the edge exists in the sparse adjacency matrix
            if adj._indices()[0][adj._indices()[1] == i].ne(j).all():
                neg_rows.append(i)
                neg_cols.append(j)
        return np.array(neg_rows), np.array(neg_cols)
    
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
    
    def fit_batch(self, epochs, adj, x,sens,idx_sens,warmup=None, adv_epoches=1):
        assert sp.issparse(adj)
        norm_w = adj.shape[0] ** 2 / float((adj.shape[0]**2 - adj.sum()) * 2)

        idx_sens = idx_sens.cpu().numpy()
        sens_mask = np.zeros((x.shape[0],1))
        sens_mask[idx_sens] = 1.0
        sens_mask = torch.from_numpy(sens_mask)

        edge_index, _ = from_scipy_sparse_matrix(adj)
        
        save_dir = "./checkpoint/{}".format(self.dataset.name)
        # check if save_dir exists, if not create it
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        miniBatchLoader = GraphSAINTRandomWalkSampler(
            Data(x=x, edge_index=edge_index, sens = sens, sens_mask = sens_mask, deg = torch.tensor(np.array(adj.sum(1)).flatten())),
                batch_size = 1000, 
                walk_length = 3,
                sample_coverage = 500, 
                num_workers = 0,
                save_dir = "./checkpoint/{}".format(self.dataset.name))

        def normalize_adjacency(adj, deg):
            # Calculate the degrees
            row, col = adj.indices()
            edge_weight = (
                adj.values() if adj.values() is not None else torch.ones(row.size(0))
            )
            # Inverse square root of degree matrix
            degree_inv_sqrt = deg.pow(-0.5)
            degree_inv_sqrt[degree_inv_sqrt == float('inf')] = 0

            # Normalize
            row_inv = degree_inv_sqrt[row]
            col_inv = degree_inv_sqrt[col]
            norm_edge_weight = edge_weight * row_inv * col_inv

            # Create the normalized sparse tensor
            adj_norm = torch.sparse.FloatTensor(torch.stack([row, col]), norm_edge_weight, adj.size())
            return adj_norm

        if warmup:
            for _ in range(warmup):
                for data in miniBatchLoader:
                    data = data.cuda()
                    sub_adj = normalize_adjacency(to_torch_sparse_tensor(data.edge_index, data.edge_norm), data.deg.float()).cuda()
                    sub_adj_dense = to_dense_adj(edge_index = data.edge_index, max_num_nodes = data.x.shape[0])[0].float()
                    adj_aug, x_aug, adj_logits = self.aug_model(sub_adj, data.x, adj_orig = sub_adj_dense)  
                    
                    edge_loss = norm_w * F.binary_cross_entropy_with_logits(adj_logits, sub_adj_dense)

                    feat_loss =  self.criterion_recons(x_aug, data.x)
                    recons_loss =  edge_loss + self.lam * feat_loss

                    self.optimizer_aug.zero_grad()
                    with torch.autograd.set_detect_anomaly(True):
                        recons_loss.backward(retain_graph=True)
                    self.optimizer_aug.step()
        
        for epoch_counter in range(epochs):
            for data in miniBatchLoader:
                data = data.cuda()

                sub_adj = normalize_adjacency(to_torch_sparse_tensor(data.edge_index, data.edge_norm), data.deg.float()).cuda()      
                sub_adj_dense = to_dense_adj(edge_index = data.edge_index, max_num_nodes = data.x.shape[0])[0].float()

                adj_aug, x_aug, adj_logits = self.aug_model(
                    sub_adj, data.x, adj_orig=sub_adj_dense
                )
                
                ### extract node representations
                h = self.projection(self.f_encoder(sub_adj, data.x))
                h_prime = self.projection(self.f_encoder(adj_aug, x_aug))

                ### update sens model
                adj_aug_nograd = adj_aug.detach()
                x_aug_nograd = x_aug.detach()

                mask = (data.sens_mask == 1.0).squeeze()

                if (epoch_counter == 0):
                    sens_epoches = adv_epoches * 10
                else:
                    sens_epoches = adv_epoches
                    
                if self.dataset.name == 'Citeseer':
                    nclasses = 6
                elif self.dataset.name == 'Cora':
                    nclasses = 7
                elif self.dataset.name == 'PubMed':
                    nclasses = 3
                else:
                    nclasses = 1

                class_counts = [sum(data.sens == i) for i in range(nclasses)]
                class_weights = [1.0 / count if count > 0 else 0 for count in class_counts]
                class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).cuda()

                for _ in range(sens_epoches):

                    s_pred , _  = self.sens_model(adj_aug_nograd, x_aug_nograd)

                    senloss = torch.nn.CrossEntropyLoss(weight=class_weights_tensor, reduction='sum')(s_pred[mask], data.sens[mask].long())

                    self.optimizer_s.zero_grad()
                    senloss.backward()
                    self.optimizer_s.step()
                
                s_pred , _  = self.sens_model(adj_aug, x_aug)
                senloss = torch.nn.CrossEntropyLoss(weight=class_weights_tensor, reduction='sum')(s_pred[mask], data.sens[mask].long())

                ## update aug model
                logits, labels = self.info_nce_loss_2views(torch.cat((h, h_prime), dim = 0))
                contrastive_loss = (nn.CrossEntropyLoss(reduction='none')(logits, labels) * data.node_norm.repeat(2)).sum()

                ## update encoder
                edge_loss = norm_w * F.binary_cross_entropy_with_logits(adj_logits, sub_adj_dense)    

                feat_loss =  self.criterion_recons(x_aug, data.x)
                recons_loss =  edge_loss + self.lam * feat_loss
                loss = self.beta * contrastive_loss + self.gamma * recons_loss - self.alpha * senloss
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                

            self.logger.log(
                {
                "loss/sens_loss": senloss.item(),
                "loss/contrastive_loss": contrastive_loss.item(),
                "loss/edge_reconstruction_loss": edge_loss.item(),
                "loss/feature_reconstruction_loss": feat_loss.item(),
                }
            )
            
            print('Epoch: {:04d}'.format(epoch_counter+1),
            'sens loss: {:.4f}'.format(senloss.item()),
            'contrastive loss: {:.4f}'.format(contrastive_loss.item()),
            'edge reconstruction loss: {:.4f}'.format(edge_loss.item()),
            'feature reconstruction loss: {:.4f}'.format(feat_loss.item()),
            )

        self.save_path = "./checkpoint/graphair_{}".format(self.dataset.name)
        torch.save(self.state_dict(),self.save_path)
        
    def fit_whole(self, epochs, adj, x,sens,idx_sens,warmup=None, adv_epoches=1):
        # print(idx_sens)
        assert sp.issparse(adj)
        if not isinstance(adj, sp.coo_matrix):
            adj = sp.coo_matrix(adj)
        adj.setdiag(1)
        adj_orig = scipysp_to_pytorchsp(adj).to_dense()
        # print how many links are in the dataset
        print("number of links in dataset:", adj_orig.sum())
        norm_w = adj_orig.shape[0]**2 / float((adj_orig.shape[0]**2 - adj_orig.sum()) * 2)
        degrees = np.array(adj.sum(1))
        degree_mat_inv_sqrt = sp.diags(np.power(degrees, -0.5).flatten())
        adj_norm = degree_mat_inv_sqrt @ adj @ degree_mat_inv_sqrt
        adj_norm = scipysp_to_pytorchsp(adj_norm)
        

        adj = adj_norm.cuda()
        
        best_contras = float("inf")

        if warmup:
            for _ in range(warmup):
                adj_aug, x_aug, adj_logits = self.aug_model(adj, x, adj_orig = adj_orig.cuda())
                edge_loss = norm_w * F.binary_cross_entropy_with_logits(adj_logits, adj_orig.cuda())

                feat_loss =  self.criterion_recons(x_aug, x)
                recons_loss =  edge_loss + self.beta * feat_loss

                self.optimizer_aug.zero_grad()
                with torch.autograd.set_detect_anomaly(True):
                    recons_loss.backward(retain_graph=True)
                self.optimizer_aug.step()

        for epoch_counter in range(epochs):
            ### generate fair view
            adj_aug, x_aug, adj_logits = self.aug_model(adj, x, adj_orig = adj_orig.cuda())

            ### extract node representations
            h = self.projection(self.f_encoder(adj, x))
            h_prime = self.projection(self.f_encoder(adj_aug, x_aug))
            # print("encoder done")

            ## update sens model
            adj_aug_nograd = adj_aug.detach()
            x_aug_nograd = x_aug.detach()
            if (epoch_counter == 0):
                sens_epoches = adv_epoches * 10
            else:
                sens_epoches = adv_epoches
                
            for _ in range(sens_epoches):
                s_pred , _  = self.sens_model(adj_aug_nograd, x_aug_nograd)
                
                senloss = self.criterion_sens(s_pred[idx_sens],sens[idx_sens].long())
                
                self.optimizer_s.zero_grad()
                senloss.backward()
                self.optimizer_s.step()
                
            s_pred , _  = self.sens_model(adj_aug, x_aug)
            senloss = self.criterion_sens(s_pred[idx_sens],sens[idx_sens].long())

            ## update aug model
            logits, labels = self.info_nce_loss_2views(torch.cat((h, h_prime), dim = 0))
            contrastive_loss = self.criterion_cont(logits, labels)

            ## update encoder
            edge_loss = norm_w * F.binary_cross_entropy_with_logits(adj_logits, adj_orig.cuda())

            feat_loss =  self.criterion_recons(x_aug, x)
            recons_loss =  edge_loss + self.lam * feat_loss
            loss = self.beta * contrastive_loss + self.gamma * recons_loss - self.alpha * senloss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            self.logger.log(
                {
                "loss/sens_loss": senloss.item(),
                "loss/contrastive_loss": contrastive_loss.item(),
                "loss/edge_reconstruction_loss": edge_loss.item(),
                "loss/feature_reconstruction_loss": feat_loss.item(),
                }
            )
            
            print('Epoch: {:04d}'.format(epoch_counter+1),
            'sens loss: {:.4f}'.format(senloss.item()),
            'contrastive loss: {:.4f}'.format(contrastive_loss.item()),
            'edge reconstruction loss: {:.4f}'.format(edge_loss.item()),
            'feature reconstruction loss: {:.4f}'.format(feat_loss.item()),
            )

        self.save_path = "./checkpoint/graphair_{}_alpha{}_beta{}_gamma{}_lambda{}".format(self.dataset, self.alpha, self.beta, self.gamma, self.lam)
        torch.save(self.state_dict(),self.save_path)
    
    # modified, we dont use the input labels now,as these are returned and specific to the link embeddings
    def test(self):
        features = self.dataset.features
        adj = self.dataset.adj
        labels = self.dataset.labels
        epochs = self.config.test_epochs
        idx_train = self.dataset.idx_train
        idx_val = self.dataset.idx_val
        idx_test = self.dataset.idx_test
        sens = self.dataset.sens
        
        features = features.cuda() if torch.cuda.is_available() else features
        h, labels, groups_mixed, groups_sub = self.forward(adj, features, sens)

        # Shuffle the embeddings and labels
        indices = torch.randperm(h.size(0))
        h = h[indices]
        labels = labels[indices]
        groups_mixed = torch.tensor(groups_mixed[indices])
        groups_sub = torch.tensor(groups_sub[indices])
        h = h.detach()

        # Move indices and labels to the correct device
        device = h.device
        idx_train = idx_train.to(device)
        idx_val = idx_val.to(device)
        idx_test = idx_test.to(device)
        labels = labels.to(device)

        acc_list = []
        roc_list = []
        dp_mixed_list = []
        eo_mixed_list = []
        dp_sub_list = []
        eo_sub_list = []

        for i in range(5):
            torch.manual_seed(i *10)
            np.random.seed(i *10)

            # train classifier
            self.classifier.reset_parameters()
                
            best_acc = best_roc = best_dp_mixed = best_eo_mixed = best_dp_sub = best_eo_sub = 0.0
            best_test_acc = best_test_roc = best_test_dp_mixed = best_test_eo_mixed = best_test_dp_sub = best_test_eo_sub = 0.0
                       
            for epoch in range(epochs):
                self.classifier.train()
                self.optimizer_classifier.zero_grad()
                output = self.classifier(h)
                loss_train = F.binary_cross_entropy_with_logits(output[idx_train], labels[idx_train].unsqueeze(1).float())
                
                loss_train.backward()
                self.optimizer_classifier.step()

                # Evaluate on validation and test sets
                self.classifier.eval()
                output = self.classifier(h)
                
                acc_val = accuracy(output[idx_val], labels[idx_val])
                acc_test = accuracy(output[idx_test], labels[idx_test])
                roc_test = auc(output[idx_test], labels[idx_test])
                
                # # Compute and print fairness metrics
                parity_val_mixed, equality_val_mixed = fair_metric(output, idx_val, labels, groups_mixed)
                parity_test_mixed, equality_test_mixed = fair_metric(output, idx_test, labels, groups_mixed)
                parity_val_sub, equality_val_sub = fair_metric(output, idx_val, labels, groups_sub)
                parity_test_sub, equality_test_sub = fair_metric(output, idx_test, labels, groups_sub)
                if epoch % 10 == 0:
                    self.logger.log(
                        {
                            f"classifier_run_{i}/": {
                                "acc_test": acc_test.item(),
                                "acc_val": acc_val.item(),
                                "dp_val_mixed": parity_val_mixed,
                                "dp_test_mixed": parity_test_mixed,
                                "eo_val_mixed": equality_val_mixed,
                                "eo_test_mixed": equality_test_mixed,
                                "dp_val_sub": parity_val_sub,
                                "dp_test_sub": parity_test_sub,
                                "eo_val_sub": equality_val_sub,
                                "eo_test_sub": equality_test_sub,
                            }
                        }
                    )

                print(
                        "Epoch [{}] Test set results:".format(epoch),
                        "acc_test= {:.4f}".format(acc_test.item()),
                        "acc_val: {:.4f}".format(acc_val.item()),
                        "dp_val: {:.4f}".format(parity_val_mixed),
                        "dp_test: {:.4f}".format(parity_test_mixed),
                        "eo_val: {:.4f}".format(equality_val_mixed),
                        "eo_test: {:.4f}".format(equality_test_mixed),
                        "dp_val: {:.4f}".format(parity_val_sub),
                        "dp_test: {:.4f}".format(parity_test_sub),
                        "eo_val: {:.4f}".format(equality_val_sub),
                        "eo_test: {:.4f}".format(equality_test_sub),
                    )

                if acc_val > best_acc:
                    best_acc = acc_val
                    best_test_acc = acc_test
                    best_test_roc = roc_test
                    best_test_dp_mixed = parity_test_mixed
                    best_test_eo_mixed = equality_test_mixed
                    best_test_dp_sub = parity_test_sub
                    best_test_eo_sub = equality_test_sub
                    
            acc_list.append(best_test_acc.detach().cpu())
            roc_list.append(best_test_roc)
            dp_mixed_list.append(best_test_dp_mixed)
            eo_mixed_list.append(best_test_eo_mixed)
            dp_sub_list.append(best_test_dp_sub)
            eo_sub_list.append(best_test_eo_sub)
        
        table = wandb.Table(
            columns=["accuracy", "accuracy_std", "roc", "roc_std", "dp_mixed", "dp_mixed_std", "eo_mixed", "eo_mixed_std", "dp_sub", "dp_sub_std", "eo_sub", "eo_sub_std"],
        )

        results = [
            np.mean(acc_list),
            np.std(acc_list),
            np.mean(roc_list),
            np.std(roc_list),
            np.mean(dp_mixed_list),
            np.std(dp_mixed_list),
            np.mean(eo_mixed_list),
            np.std(eo_mixed_list),
            np.mean(dp_sub_list),
            np.std(dp_sub_list),
            np.mean(eo_sub_list),
            np.std(eo_sub_list),
        ]
        
        table.add_data(*results)
        self.logger.log({"table/results": table})
        
        self.logger.log(
            {
                "results/accuracy": np.mean(acc_list),
                "results/roc": np.mean(roc_list),
                "results/dp_mixed": np.mean(dp_mixed_list),
                "results/eo_mixed": np.mean(eo_mixed_list),
                "results/dp_sub": np.mean(dp_sub_list),
                "results/eo_sub": np.mean(eo_sub_list),
            }
        )
           
        # Print average results
        print("Avg results:",
            "acc: {:.4f} std: {:.4f}".format(np.mean(acc_list), np.std(acc_list)),
            "AUC: {:.4f} std: {:.4f}".format(np.mean(roc_list), np.std(roc_list)),
            "dp-mixed: {:.4f} std: {:.4f}".format(np.mean(dp_mixed_list), np.std(dp_mixed_list)),
            "dp-sub: {:.4f} std: {:.4f}".format(np.mean(dp_sub_list), np.std(dp_sub_list)),
            "eo-mixed: {:.4f} std: {:.4f}".format(np.mean(eo_mixed_list), np.std(eo_mixed_list)),
            "eo-sub: {:.4f} std: {:.4f}".format(np.mean(eo_sub_list), np.std(eo_sub_list)))

        return {'accuracy': [np.mean(acc_list), np.std(acc_list)],
                'AUC': [np.mean(roc_list), np.std(roc_list)],
                'dp-mixed': [np.mean(dp_mixed_list), np.std(dp_mixed_list)],
                'eo-mixed': [np.mean(eo_mixed_list), np.std(eo_mixed_list)],
                'dp-sub': [np.mean(dp_sub_list), np.std(dp_sub_list)],
                'eo-sub': [np.mean(eo_sub_list), np.std(eo_sub_list)]}