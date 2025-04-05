import os
from functools import partial
import inspect

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import VisionDataset

import numpy as np
from sklearn.base import BaseEstimator
from .torch_utils.losses import get_outputs_loss
from .torch_utils import get_optimizer, get_loss, get_scheduler, CustomTensorDataset
from .torch_utils import archs, data_augs

DEBUG = int(os.getenv("DEBUG", 0))


class TorchModelBase(BaseEstimator):

    def _get_dataset(self, X, y=None, sample_weights=None, is_img_data=True):
        X = self._preprocess_x(X, is_img_data=is_img_data)
        if sample_weights is None:
            sample_weights = np.ones(len(X))

        if self.dataaug is None:
            transform = None
        else:
            if y is None:
                transform = getattr(data_augs, self.dataaug)()[1]
            else:
                transform = getattr(data_augs, self.dataaug)()[0]

        if y is None:
            return CustomTensorDataset((torch.from_numpy(X).float(), ), transform=transform)
        if 'mse' in self.loss_name:
            Y = self.lbl_enc.transform(y.reshape(-1, 1))
            dataset = CustomTensorDataset(
                (torch.from_numpy(X).float(), torch.from_numpy(Y).float(), torch.from_numpy(sample_weights).float()), transform=transform)
        else:
            dataset = CustomTensorDataset(
                (torch.from_numpy(X).float(), torch.from_numpy(y).long(), torch.from_numpy(sample_weights).float()), transform=transform)
        return dataset

    def _calc_eval(self, loader, loss_fn):
        cum_loss, cum_acc = 0., 0.
        with torch.no_grad():
            for data in loader:
                tx, ty = data[0], data[1]
                tx, ty = tx.to(self.device), ty.to(self.device)
                outputs = self.model(tx)
                if loss_fn.reduction == 'none':
                    loss = torch.sum(loss_fn(outputs, ty))
                else:
                    loss = loss_fn(outputs, ty)
                cum_loss += loss.item()
                cum_acc += (outputs.argmax(dim=1)==ty).sum().float().item()
        return cum_loss / len(loader.dataset), cum_acc / len(loader.dataset)

    def _preprocess_x(self, X, is_img_data=True):
        if len(X.shape) == 4 and is_img_data == True:
            return X.transpose(0, 3, 1, 2)
        else:
            return X

    def fit(self, X, y, sample_weights=None, verbose=None, is_img_data=True):
        dataset = self._get_dataset(X, y, sample_weights, is_img_data=is_img_data)
        return self.fit_dataset(dataset, verbose=verbose)

    def _prep_pred(self, X, is_img_data=True):
        self.model.eval()
        if isinstance(X, VisionDataset):
            dataset = X
        else:
            if self.dataaug is None:
                transform = None
            else:
                transform = getattr(data_augs, self.dataaug)()[1]
            X = self._preprocess_x(X, is_img_data=is_img_data)
            dataset = CustomTensorDataset((torch.from_numpy(X).float(), ), transform=transform)
        loader = torch.utils.data.DataLoader(dataset,
            batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        return loader

    def predict_ds(self, ds):
        loader = torch.utils.data.DataLoader(ds,
            batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        ret = []
        for x in loader:
            x = x[0]
            ret.append(self.model(x.to(self.device)).argmax(1).cpu().numpy())
        del loader
        return np.concatenate(ret)

    def get_repr(self, X, is_img_data=True):
        loader = self._prep_pred(X, is_img_data=is_img_data)
        ret = []
        for [x] in loader:
            ret.append(self.model.get_repr(x.to(self.device)).detach().cpu().numpy())
        del loader
        return np.concatenate(ret)

    def predict(self, X, is_img_data=True):
        loader = self._prep_pred(X, is_img_data=is_img_data)
        ret = []
        for [x] in loader:
            x.requires_grad_(False)
            ret.append(self.model(x.to(self.device)).argmax(1).cpu().numpy())
        del loader
        return np.concatenate(ret)

    def predict_proba(self, X, is_img_data=True):
        loader = self._prep_pred(X, is_img_data=is_img_data)
        ret = []
        for [x] in loader:
            x.requires_grad_(False)
            output = F.softmax(self.model(x.to(self.device)).detach(), dim=1)
            ret.append(output.cpu().numpy())
        del loader
        return np.concatenate(ret, axis=0)

    def predict_real(self, X, is_img_data=True):
        loader = self._prep_pred(X, is_img_data=is_img_data)
        ret = []
        for [x] in loader:
            x.requires_grad_(False)
            ret.append(self.model(x.to(self.device)).detach().cpu().numpy())
        del loader
        return np.concatenate(ret, axis=0)

    def save(self, path):
        if self.multigpu:
            model_state_dict = self.model.module.state_dict()
        else:
            model_state_dict = self.model.state_dict()
        torch.save({
            'epoch': self.start_epoch,
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path % self.start_epoch)

    def load(self, path):
        loaded = torch.load(path)
        if 'epoch' in loaded:
            self.start_epoch = loaded['epoch']
            self.model.load_state_dict(loaded['model_state_dict'])
            self.optimizer.load_state_dict(loaded['optimizer_state_dict'])
        else:
            self.model.load_state_dict(loaded)
        self.model.eval()

    def dset_pred_and_lbl(self, ds):
        loader = torch.utils.data.DataLoader(ds,
            batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        preds, lbls = [], []
        for (x, y) in loader:
            pred = self.model(x.to(self.device)).argmax(1).cpu().numpy()
            preds.append(pred)
            lbls.append(y.numpy())
        del loader
        return np.concatenate(preds), np.concatenate(lbls)