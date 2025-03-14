import logging

import numpy as np
import torch
import wandb
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from torch import trace, inverse
from scipy.linalg import fractional_matrix_power

from FedML.fedml_core.trainer.model_trainer import ModelTrainer


# Trainer for MoleculeNet. The evaluation metric is ROC-AUC

class GatMoleculeNetTrainer(ModelTrainer):

    def get_model_params(self):
        return self.model.cpu().state_dict()

    def set_model_params(self, model_parameters):
        logging.info("set_model_params")
        self.model.load_state_dict(model_parameters)

    def train(self, train_data, device, args):
        model = self.model
        if args.is_mtl:
            self.model.omega_corr.to(device)
        model.to(device)
        model.train()

        test_data = None
        try:
            test_data = self.test_data
        except:
            pass

        criterion = torch.nn.BCEWithLogitsLoss(reduction='none')
        if args.client_optimizer == "sgd":
            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
            # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay = args.wd)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
            # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay = args.wd)


        max_test_score = 0
        best_model_params = {}
        eps = 1e-5
        for epoch in range(args.epochs):
            for mol_idxs, (adj_matrix, feature_matrix, label, mask , cli_mask) in enumerate(train_data):
                # Pass on molecules that have no labels
                mask = mask.to(device=device, dtype=torch.float32, non_blocking=True)
                cli_mask = cli_mask.to(device=device, dtype=torch.float32, non_blocking=True) if cli_mask is not None else None
                mask = mask * cli_mask if cli_mask is not None else mask
                if torch.all(mask == 0).item():
                    continue

                optimizer.zero_grad()

                adj_matrix = adj_matrix.to(device=device, dtype=torch.float32, non_blocking=True)
                feature_matrix = feature_matrix.to(device=device, dtype=torch.float32, non_blocking=True)
                label = label.to(device=device, dtype=torch.float32, non_blocking=True)

                logits = model(adj_matrix, feature_matrix)
                clf_loss = criterion(logits, label) * mask
                clf_loss = clf_loss.sum() / mask.sum()

                if args.is_mtl:
                    W = self.model.readout.output.weight
                    lhs = torch.mm(W.t() , torch.inverse(self.model.omega_corr + eps * torch.eye(self.model.omega_corr.shape[0], 
                                                                                      dtype=self.model.omega_corr.dtype, device=self.model.omega_corr.device)))
                    trace_in = torch.mm(lhs, W)
                    loss = clf_loss +  self.task_reg * torch.trace(trace_in) + args.wd * 0.5 * torch.norm(W)**2
                    loss.backward()
                    optimizer.step()
                    with torch.no_grad():
                        W = self.model.readout.output.weight
                        mul = torch.mm(W, W.t())
                        fr_pow = fractional_matrix_power(mul.cpu(), 1/2)
                        self.model.omega_corr = torch.nn.Parameter(torch.Tensor(fr_pow / np.trace(fr_pow)).to(device))
                        self.model.omega_corr.requires_grad=False   
                else:
                    clf_loss.backward()
                    optimizer.step()                             

                if ((mol_idxs + 1) % args.frequency_of_the_test == 0) or (mol_idxs == len(train_data) - 1):
                    if test_data is not None:
                        test_score, _ = self.test(self.test_data, device, args)
                        print('Epoch = {}, Iter = {}/{}: Test Score = {}'.format(epoch, mol_idxs + 1, len(train_data), test_score))
                        if test_score > max_test_score:
                            max_test_score = test_score
                            best_model_params = {k: v.cpu() for k, v in model.state_dict().items()}
                        print('Current best = {}'.format(max_test_score))
            
            self.task_reg *= args.task_reg_decay

        return max_test_score, best_model_params



    def test(self, test_data, device, args):
        logging.info("----------test--------")
        model = self.model
        model.eval()
        model.to(device)

        with torch.no_grad():
            y_pred = []
            y_true = []
            masks = []
            for mol_idx, (adj_matrix, feature_matrix, label, mask , _) in enumerate(test_data):
                adj_matrix = adj_matrix.to(device=device, dtype=torch.float32, non_blocking=True)
                feature_matrix = feature_matrix.to(device=device, dtype=torch.float32, non_blocking=True)

                logits = model(adj_matrix, feature_matrix)

                y_pred.append(logits.cpu().numpy())
                y_true.append(label.cpu().numpy())
                masks.append(mask.numpy())

        y_pred = np.array(y_pred)
        y_true = np.array(y_true)
        masks = np.array(masks)

        results = []
        for label in range(masks.shape[1]):
            valid_idxs = np.nonzero(masks[:, label])
            truth = y_true[valid_idxs, label].flatten()
            pred = y_pred[valid_idxs, label].flatten()

            if np.all(truth == 0.0) or np.all(truth == 1.0):
                results.append(float('nan'))
            else:
                if args.metric == 'prc-auc':
                    precision, recall, _ = precision_recall_curve(truth, pred)
                    score = auc(recall, precision)
                else:
                    score = roc_auc_score(truth, pred)

                results.append(score)

        score = np.nanmean(results)

        return score, model

    def test_on_the_server(self, train_data_local_dict, test_data_local_dict, device, args=None) -> bool:
        logging.info("----------test_on_the_server--------")

        model_list, score_list = [], []
        for client_idx in test_data_local_dict.keys():
            test_data = test_data_local_dict[client_idx]
            score, model = self.test(test_data, device, args)
            for idx in range(len(model_list)):
                self._compare_models(model, model_list[idx])
            model_list.append(model)
            score_list.append(score)
            if args.dataset != "pcba":
                logging.info('Client {}, Test ROC-AUC score = {}'.format(client_idx, score))
                wandb.log({"Client {} Test/ROC-AUC".format(client_idx): score})
            else:
                logging.info('Client {}, Test PRC-AUC score = {}'.format(client_idx, score))
                wandb.log({"Client {} Test/PRC-AUC".format(client_idx): score})
        avg_score = np.mean(np.array(score_list))
        if args.dataset != "pcba":
            logging.info('Test ROC-AUC Score = {}'.format(avg_score))
            wandb.log({"Test/ROC-AUC": avg_score})
        else:
            logging.info('Test PRC-AUC Score = {}'.format(avg_score))
            wandb.log({"Test/PRC-AUC": avg_score})
        return True

    def _compare_models(self, model_1, model_2):
        models_differ = 0
        for key_item_1, key_item_2 in zip(model_1.state_dict().items(), model_2.state_dict().items()):
            if torch.equal(key_item_1[1], key_item_2[1]):
                pass
            else:
                models_differ += 1
                if key_item_1[0] == key_item_2[0]:
                    logging.info('Mismtach found at', key_item_1[0])
                else:
                    raise Exception
        if models_differ == 0:
            logging.info('Models match perfectly! :)')
