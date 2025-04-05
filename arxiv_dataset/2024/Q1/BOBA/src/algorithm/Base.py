"""
An Abstract Base Class of Federated Learning
"""

import torch
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

from model import create_model, create_loss, create_metric, create_optimizer
from utils import History


class BaseServer:
    """
    Base Class of Server
    """

    def __init__(self, client_datasets, server_datasets, test_dataset, args):
        # some useful information
        self.num_clients = len(client_datasets)
        self.idx2cid = {i: cid for i, cid in enumerate(client_datasets)}
        self.cid2idx = {cid: i for i, cid in self.idx2cid.items()}

        self.gm_rounds = args.gm_rounds

        # server dataloaders
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.device = args.device

        # server dataset (usually for training or evaluating model)
        self.server_datasets = server_datasets
        self.server_dataloaders = {}
        for key, dataset in self.server_datasets.items():
            self.server_dataloaders[key] = DataLoader(dataset, batch_size=self.batch_size, shuffle=True,
                                                      drop_last=False, num_workers=self.num_workers)

        # test dataset (This is NOT used in training! )
        self.test_dataset = test_dataset
        self.test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, drop_last=False,
                                          num_workers=self.num_workers)

        # history
        self.history = History()
        self.history.concat({
            'idx2cid': self.idx2cid,
            'cid2idx': self.cid2idx,
        })

    def run(self, args):
        """
        Only use server data for training
        :param args:
        :return:
        """

        # model
        self.model = create_model(args)

        for rnd in range(1, self.gm_rounds + 1):
            tqdm.write('Round: %d / %d' % (rnd, self.gm_rounds))
            self.server_train(self.model, args)
            avg_loss, avg_metric, total_examples = self.test(self.model, args)
            print(avg_metric)

    def server_train(self, model, args, vector=None, dataset='all', return_vector=False):
        """
        Use Server data for training (one round)
        :param model:
        :param args:
        :param dataset: use which part of the server data to train
        :return:
        """

        # ======== ======== Extract Hyperparameters ======== ========
        loss_func = create_loss(args.loss)
        metric_func = create_metric(args.metric)
        optimizer = create_optimizer(model, args.lm_opt, args.lm_lr)

        if vector is not None:
            model.load_params_tensor(vector)

        # ======== ======== Training ======== ========
        model.train()

        total_examples, total_loss, total_metric = 0, 0, 0

        for *X, Y in self.server_dataloaders[dataset]:
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

        avg_loss, avg_metric = total_loss / total_examples, total_metric / total_examples

        if return_vector:
            return model.get_params_tensor()
        else:
            return avg_loss, avg_metric, total_examples

    def server_eval(self, model, args, vector=None, dataset='all'):
        """
        Use server data to eval model
        :return:
        """
        loss_func = create_loss(args.loss)
        metric_func = create_metric(args.metric)

        if vector is not None:
            model.load_params_tensor(vector)

        model.eval()

        total_examples, total_loss, total_metric = 0, 0, 0

        with torch.no_grad():
            for *X, Y in self.server_dataloaders[dataset]:
                # Get a batch of data
                X = [x.to(self.device) for x in X]
                Y = Y.to(self.device)

                # get prediction
                logits = model(*X)
                loss = loss_func(logits, Y)
                metric = metric_func(logits, Y)

                num_examples = len(X[0])
                total_examples += num_examples

                total_loss += loss.item() * num_examples

                total_metric += metric.item() * num_examples

        avg_loss, avg_metric = total_loss / total_examples, total_metric / total_examples

        return avg_loss, avg_metric, total_examples

    def test(self, model, args):
        """
        Test the performance of the model.
        :return:
        """
        loss_func = create_loss(args.loss)
        metric_func = create_metric(args.metric)

        model.eval()

        total_examples, total_loss, total_metric = 0, 0, 0

        all_pred = []
        all_targets = []

        with torch.no_grad():
            for *X, Y in self.test_dataloader:
                # Get a batch of data
                X = [x.to(self.device) for x in X]
                Y = Y.to(self.device)

                # get prediction
                logits = model(*X)
                loss = loss_func(logits, Y)
                metric = metric_func(logits, Y)

                num_examples = len(X[0])
                total_examples += num_examples

                total_loss += loss.item() * num_examples

                total_metric += metric.item() * num_examples

                # save all predictions
                pred = logits.argmax(dim=1)
                all_pred += list(pred.detach().cpu().numpy())
                all_targets += list(Y.detach().cpu().numpy())

        avg_loss, avg_metric = total_loss / total_examples, total_metric / total_examples
        cm = confusion_matrix(all_targets, all_pred)

        tqdm.write('\t Test:  Loss: %.4f \t Metric: %.4f' % (avg_loss, avg_metric))

        log_dict = {
            'GM_test_avg_loss': avg_loss,
            'GM_test_avg_metric': avg_metric,
            'GM_test_confusion_matrix': cm,
        }
        self.history.append(log_dict)

        return avg_loss, avg_metric, total_examples


class BaseClient:
    """
    Base Class of Client
    """

    def __init__(self, cid, datasets, args):
        self.cid = cid

        # client local dataset
        self.datasets = datasets  # e.g. ['train', 'test'] or ['train', 'val', 'test']

        # number of training data / testing data / ...
        self.num_data = {key: len(dataset) for key, dataset in datasets.items()}
        self.num_data['all'] = sum([len(dataset) for dataset in datasets.values()])

        # client local dataloaders
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.device = args.device

        self.dataloaders = {}
        for key, dataset in self.datasets.items():
            if key in ['train', ]:
                # for training set, we shuffle the data
                self.dataloaders[key] = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=False,
                                                   num_workers=self.num_workers)
            elif key in ['valid', 'test', ]:
                # for testing set, we need to evaluate every data points. Also it is not necessary to shuffle.
                self.dataloaders[key] = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, drop_last=False,
                                                   num_workers=self.num_workers)

    def local_train(self, model, args, dataset='train'):
        """
        Local Training
        (Just a template. )
        """
        model.train()

        avg_loss, avg_metric = np.inf, 0.0
        num_data = self.num_data[dataset]

        return avg_loss, avg_metric, num_data

    def local_eval(self, model, args, dataset='test'):
        """
        Local Evaluation
        """
        loss_func = create_loss(args.loss)
        metric_func = create_metric(args.metric)

        model.eval()

        total_examples, total_loss, total_metric = 0, 0, 0

        with torch.no_grad():
            for *X, Y in self.dataloaders[dataset]:
                # Get a batch of data
                X = [x.to(self.device) for x in X]
                Y = Y.to(self.device)

                # get prediction
                logits = model(*X)
                loss = loss_func(logits, Y)
                metric = metric_func(logits, Y)

                num_examples = len(X[0])
                total_examples += num_examples

                total_loss += loss.item() * num_examples

                total_metric += metric.item() * num_examples

        avg_loss, avg_metric = total_loss / total_examples, total_metric / total_examples

        return avg_loss, avg_metric, total_examples
