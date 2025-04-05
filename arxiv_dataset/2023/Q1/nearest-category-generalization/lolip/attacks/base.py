from abc import abstractmethod

import torch
import numpy as np


class AttackModel():

    def __init__(self, norm):
        self.norm = norm
        super().__init__()

    def _preprocess_x(self, X, is_img_data=True):
        if len(X.shape) == 4 and is_img_data:
            return torch.from_numpy(X.transpose(0, 3, 1, 2)).float()
        else:
            return torch.from_numpy(X).float()

    def _pert_with_eps_constraint(self, pert_X, eps):
        if isinstance(eps, list):
            rret = []
            norms = np.linalg.norm(pert_X, axis=1, ord=self.norm)
            for ep in eps:
                t = np.copy(pert_X)
                t[norms > ep, :] = 0
                rret.append(t)
            return rret
        elif eps is not None:
            pert_X[np.linalg.norm(pert_X, axis=1, ord=self.norm) > eps, :] = 0
            return pert_X
        else:
            return pert_X

    @abstractmethod
    def perturb_ds(self, ds, eps):
        pass

    @abstractmethod
    def perturb(self, X, y, eps):
        pass


class TorchAttackModel(AttackModel):

    def __init__(self, model_fn, norm, batch_size, device=None):
        super().__init__(norm=norm)
        self.model_fn = model_fn
        self.batch_size = batch_size
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

    def _get_loader(self, X, y, is_img_data=True):
        dataset = torch.utils.data.TensorDataset(
            self._preprocess_x(X, is_img_data=is_img_data), torch.from_numpy(y).long())
        loader = torch.utils.data.DataLoader(dataset,
            batch_size=self.batch_size, shuffle=False, num_workers=1)
        return loader