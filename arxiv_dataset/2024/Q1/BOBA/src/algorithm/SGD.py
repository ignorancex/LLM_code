"""
SGD (Federated Averaging)

Reference:
    Brendan McMahan, Eider Moore, Daniel Ramage, Seth Hampson, Blaise Agüera y Arcas:
    Communication-Efficient Learning of Deep Networks from Decentralized Data. AISTATS 2017: 1273-1282
Implementation:
    https://github.com/pliang279/LG-SGD
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from copy import deepcopy

from model import create_model, create_loss, create_metric, create_optimizer

from attack import create_attack
from defense import create_defense

from .Base import BaseServer, BaseClient


class SGDServer(BaseServer):
    """
    Server of SGD
    """

    def __init__(self, client_datasets, server_dataset, test_dataset, args):
        super(SGDServer, self).__init__(client_datasets, server_dataset, test_dataset, args)

        # check or set hyperparameters
        assert args.gm_opt == 'sgd'
        assert args.gm_lr == 1.0
        self.gm_rounds = args.gm_rounds

        # sample a subset of clients per communication round
        self.cohort_size = max(1, round(self.num_clients * args.part_rate))

        # init clients
        self.clients = {cid: SGDClient(cid, datasets, args) for cid, datasets in client_datasets.items()}

        # init byzantine
        self.attacker = create_attack(args)

        # init aggregator
        self.aggregator = create_defense(args)

        # model
        self.model = create_model(args)

        # SGD is not personalized federated learning
        self.is_pfl = False

    def run(self, args):
        """
        Run the training and testing pipeline
        """

        for rnd in range(1, self.gm_rounds + 1):
            tqdm.write('Round: %d / %d' % (rnd, self.gm_rounds))
            self.train(self.model, args)
            self.test(self.model, args)

            # lr decay
            if rnd >= args.decay_start_round and rnd % args.decay_per_round == 0:
                args.lm_lr = args.lm_lr * args.decay_rate

    def train(self, model, args):
        """
        Train for one communication round
        """
        # current global model
        global_state = deepcopy(model.uploaded_state_dict())
        origin = model.get_params_tensor()

        tensors = []  # local model parameters
        weights = []  # weights (importance) for each client
        losses = []  # training losses for local models (LMs)
        metrics = []  # training metrics (accuracies) for local models (LMs)

        # sample a subset of clients
        selected_idxs = sorted(list(torch.randperm(self.num_clients)[:self.cohort_size].numpy()))
        selected_cids = [self.idx2cid[idx] for idx in selected_idxs]

        # iterate randomly selected honest clients
        for cid in tqdm(selected_cids):
            client = self.clients[cid]
            model.load_state_dict(global_state, strict=False)  # start from global model

            loss, metric, num_data = client.local_train(model, args, 'train')
            tensor = model.get_params_tensor()

            tensors.append(tensor)
            weights.append(num_data)
            losses.append(loss)
            metrics.append(metric)

        # train loss and metric
        agg_loss = sum([weight * loss for weight, loss in zip(weights, losses)]) / sum(weights)
        agg_metric = sum([weight * metric for weight, metric in zip(weights, metrics)]) / sum(weights)
        tqdm.write('\t Train: Loss: %.4f \t Metric: %.4f' % (agg_loss, agg_metric))

        log_dict = {
            'train_selected_idxs': selected_idxs,
            'train_selected_cids': selected_cids,
            'LM_train_losses': losses,
            'LM_train_metrics': metrics,
            'LM_train_wavg_loss': agg_loss,
            'LM_train_wavg_metric': agg_metric,
        }
        self.history.append(log_dict)

        # attack
        if args.num_byz > 0:
            hon_matrix = torch.stack(tensors)
            model.load_state_dict(global_state, strict=False)
            byz_matrix = self.attacker.attack(model, hon_matrix)

            matrix = torch.cat([hon_matrix, byz_matrix], dim=0)

        else:
            matrix = torch.stack(tensors)

        # model aggregation
        if self.aggregator.server_todo is None:
            agg_vector = self.aggregator.aggregate(matrix, origin)

        elif self.aggregator.server_todo == 'vector':
            model.load_state_dict(global_state, strict=False)
            self.server_train(model, args, dataset='all')
            server_vector = model.get_params_tensor()
            agg_vector = self.aggregator.aggregate(matrix, origin, server_vector)

        elif self.aggregator.server_todo == 'matrix':
            server_matrix = []
            for i in range(args.num_labels):
                model.load_state_dict(global_state, strict=False)
                self.server_train(model, args, dataset=i)
                server_matrix.append(model.get_params_tensor())
            server_matrix = torch.stack(server_matrix)
            agg_vector = self.aggregator.aggregate(matrix, origin, server_matrix)

        elif self.aggregator.server_todo == 'train':

            train_func = lambda vector: self.server_train(model, args, vector=vector, dataset='all', return_vector=True)
            agg_vector = self.aggregator.aggregate(matrix, origin, train_func)

        elif self.aggregator.server_todo == 'eval':

            # model.load_state_dict(global_state, strict=False)
            eval_func = lambda vector: self.server_eval(model, args, vector=vector, dataset='all')
            agg_vector = self.aggregator.aggregate(matrix, origin, eval_func)

        else:
            raise NotImplementedError

        if args.server_gradient_weight != 0.0:
            model.load_state_dict(global_state, strict=False)
            server_vector = self.server_train(model, args, dataset='all', return_vector=True)
            agg_vector = args.server_gradient_weight * server_vector + (1 - args.server_gradient_weight) * agg_vector

        model.load_params_tensor(agg_vector)

    def eval(self, model, args):
        """
        Evaluate the global model
        """
        weights = []  # weights (importance) for each client
        losses = []  # local testing losses
        metrics = []  # local testing metrics (accuracies)

        for cid, client in tqdm(self.clients.items()):
            loss, metric, num_data = client.local_eval(model, args, 'test')
            weights.append(num_data)
            losses.append(loss)
            metrics.append(metric)

        # eval loss and metric
        agg_loss = sum([weight * loss for weight, loss in zip(weights, losses)]) / sum(weights)
        agg_metric = sum([weight * metric for weight, metric in zip(weights, metrics)]) / sum(weights)
        tqdm.write('\t Eval:  Loss: %.4f \t Metric: %.4f' % (agg_loss, agg_metric))

        log_dict = {
            'GM_test_losses': losses,
            'GM_test_metrics': metrics,
            'GM_test_wavg_loss': agg_loss,
            'GM_test_wavg_metric': agg_metric,
        }
        self.history.append(log_dict)


class SGDClient(BaseClient):
    """
    Client of SGD
    """

    def __init__(self, cid, datasets, args):
        super(SGDClient, self).__init__(cid, datasets, args)

    def local_train(self, model, args, dataset='train'):
        """
        Local Training
        """

        # ======== ======== Extract Hyperparameters ======== ========
        loss_func = create_loss(args.loss)
        metric_func = create_metric(args.metric)
        optimizer = create_optimizer(model, args.lm_opt, args.lm_lr)
        num_epochs = args.lm_epochs

        # ======== ======== Prepare for Training ======== ========
        dataloader = self.dataloaders[dataset]
        num_data = self.num_data[dataset]

        # ======== ======== Training ======== ========
        model.train()

        total_examples, total_loss, total_metric = 0, 0, 0

        for epoch in range(num_epochs):
            for *X, Y in dataloader:
                # Get a batch of data
                X = [x.to(self.device) for x in X]
                Y = Y.to(self.device)

                # get prediction
                logits = model(*X)
                loss = loss_func(logits, Y)

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                with torch.no_grad():
                    # record the loss and accuracy
                    num_examples = len(X[0])
                    total_examples += num_examples

                    total_loss += loss.item() * num_examples

                    metric = metric_func(logits, Y)
                    total_metric += metric.item() * num_examples

                break  # only one epoch


        avg_loss, avg_metric = total_loss / total_examples, total_metric / total_examples

        return avg_loss, avg_metric, num_data
