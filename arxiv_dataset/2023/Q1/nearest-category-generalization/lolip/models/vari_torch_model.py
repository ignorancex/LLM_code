import gc
import os
import inspect

import torch
from tqdm import tqdm
from torchvision.datasets import VisionDataset
import numpy as np
from mkdir_p import mkdir_p

from .torch_utils.losses import get_outputs_loss
from .torch_utils import get_optimizer, get_loss, get_scheduler, CustomTensorDataset
from .torch_utils import archs, data_augs
from .base import TorchModelBase
from ..tools.nn_utils import get_region_proj_fn_list, get_ellipsoid_proj_fn_list
from ..tools.nn_utils import get_nearest_oppo_dist
from ..tools.nn_utils.pca_ellipsoid import get_pca_ellipsoid_proj_fn_list
from ..tools.nn_utils.batch_sub_voronoi import get_sub_voronoi_info_list

DEBUG = int(os.getenv("DEBUG", 0))

class VariTorchModel(TorchModelBase):
    def __init__(self, lbl_enc, n_features, n_classes, loss_name='ce',
                n_channels=None, learning_rate=1e-4, momentum=0.0, weight_decay=0.0,
                batch_size=256, epochs=20, optimizer='sgd', architecture='arch_001',
                random_state=None, eval_callbacks=None, train_type=None, eps: float=0.1,
                norm=np.inf, multigpu=False, dataaug=None, device=None, num_workers=4):
        print(f'lr: {learning_rate}, opt: {optimizer}, loss: {loss_name}, '
              f'arch: {architecture}, dataaug: {dataaug}, batch_size: {batch_size}, '
              f'momentum: {momentum}, weight_decay: {weight_decay}, eps: {eps}, '
              f'epochs: {epochs}, train_type: {train_type}')
        self.num_workers = num_workers
        self.n_features = n_features
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.architecture = architecture
        self.epochs = epochs
        self.lbl_enc = lbl_enc
        self.loss_name = loss_name
        self.dataaug = dataaug

        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        arch_fn = getattr(archs, self.architecture)
        arch_params = dict(n_classes=self.n_classes, n_channels=n_channels)
        if 'n_features' in inspect.getfullargspec(arch_fn)[0]:
            arch_params['n_features'] = n_features

        model = arch_fn(**arch_params)
        if self.device == 'cuda':
            model = model.cuda()

        self.multigpu = multigpu
        if self.multigpu:
            model = torch.nn.DataParallel(model, device_ids=[0, 1])

        self.optimizer = get_optimizer(model, optimizer, learning_rate, momentum, weight_decay)
        self.model = model

        self.eval_callbacks = eval_callbacks
        self.random_state = random_state
        self.train_type = train_type

        self.tst_ds = None
        self.start_epoch = 1

        self.current_eps = None

        ### Attack ####
        self.eps = eps
        self.norm = norm
        ###############

    def _get_dataset(self, X, y=None, training=True, sample_weights=None,
                     radius=None, is_img_data=True):
        X = self._preprocess_x(X, is_img_data=is_img_data)
        if sample_weights is None:
            sample_weights = np.ones(len(X))
        if radius is None:
            radius = np.ones(len(X)) * self.eps

        if self.dataaug is None:
            transform = None
        else:
            if training == False:
                transform = getattr(data_augs, self.dataaug)()[1]
            else:
                transform = getattr(data_augs, self.dataaug)()[0]

        if training == False:
            if y is None:
                return CustomTensorDataset((torch.from_numpy(X).float(), ), transform=transform)
            else:
                return CustomTensorDataset((torch.from_numpy(X).float(), torch.from_numpy(y).long()), transform=transform)

        if not callable(radius[0]):
            radius = torch.from_numpy(radius).float()
        else:
            radius = torch.from_numpy(np.arange(len(radius))).long()

        if 'mse' in self.loss_name:
            Y = self.lbl_enc.transform(y.reshape(-1, 1))
            dataset = CustomTensorDataset(
                (torch.from_numpy(X).float(), torch.from_numpy(Y).float(),
                 torch.from_numpy(sample_weights).float(), radius), transform=transform)
        else:
            dataset = CustomTensorDataset(
                (torch.from_numpy(X).float(), torch.from_numpy(y).long(),
                 torch.from_numpy(sample_weights).float(), radius), transform=transform)
        return dataset

    def _get_nearest_oppo_dist(self, X, y, cache_filename=None):
        return get_nearest_oppo_dist(X, y, self.norm, cache_filename=cache_filename)

    def fit(self, X, y, unX=None, sample_weights=None, idx_cache_filename=None,
            cache_filename=None, is_img_data=True, verbose=None,
            normalized=False):
        """
        X, y: nparray
        """
        X = np.asarray(X, dtype=np.float32)
        self.eps_x = np.ones(len(X)) * self.eps
        if unX is None:
            unlabeled_ds = None
            if self.train_type is None:
                pass
            elif "halfclose" in self.train_type:
                #self.eps_x = self._get_nearest_oppo_dist(torch.from_numpy(X).float(), y) / 2.
                self.eps_x = self._get_nearest_oppo_dist(X, y, cache_filename=cache_filename) / 2.
                if "halfclose10" in self.train_type:
                    self.eps_x = self.eps_x / 10.
                elif "halfclose.75" in self.train_type:
                    self.eps_x = self.eps_x / .75
                elif "halfclose.5" in self.train_type:
                    self.eps_x = self.eps_x / .5
                elif "halfclose20" in self.train_type:
                    self.eps_x = self.eps_x / 20.
                elif "halfclose50" in self.train_type:
                    self.eps_x = self.eps_x / 50.
                elif "halfclose2" in self.train_type:
                    self.eps_x = self.eps_x / 2.
                elif "halfclose3" in self.train_type:
                    self.eps_x = self.eps_x / 3.
                elif "halfclose1.05" in self.train_type:
                    self.eps_x = self.eps_x / 1.05
                elif "halfclose5" in self.train_type:
                    self.eps_x = self.eps_x / 5.

            elif "batchsubvor" in self.train_type:
                if "batchsubvor20" in self.train_type:
                    n_samples = 20
                elif "batchsubvor50" in self.train_type:
                    n_samples = 50
                elif "batchsubvor100" in self.train_type:
                    n_samples = 100
                elif "batchsubvor200" in self.train_type:
                    n_samples = 200
                else:
                    n_samples = -1

                if "rand" in self.train_type:
                    sample_type = "rand"
                else:
                    sample_type = "closest"

                if "S75" in self.train_type:
                    scale_h = 0.75
                elif "S50" in self.train_type:
                    scale_h = 0.50
                else:
                    scale_h = 1

                est_stepsize = False
                if "ada" in self.train_type:
                    est_stepsize = True

                self.eps_x = get_sub_voronoi_info_list(X, y, scale_h=scale_h,
                    sample_type=sample_type, n_samples=n_samples, est_stepsize=est_stepsize,
                    n_jobs=4, random_state=self.random_state)

            elif "1nnregion" in self.train_type:
                if "1nnregion20" in self.train_type:
                    n_samples = 20
                elif "1nnregion50" in self.train_type:
                    n_samples = 50
                else:
                    n_samples = -1
                self.eps_x = get_region_proj_fn_list(X, y, self.norm,
                        n_samples=n_samples, n_jobs=4, random_state=self.random_state)

            elif "maxellip" in self.train_type:
                if "maxellip20" in self.train_type:
                    n_samples = 20
                elif "maxellip50" in self.train_type:
                    n_samples = 50
                else:
                    n_samples = -1
                self.eps_x = get_ellipsoid_proj_fn_list(X, y, self.norm, n_samples=n_samples,
                        n_jobs=4, random_state=self.random_state, cache_filename=cache_filename)

            elif "pcaellip" in self.train_type:
                if "T10" in self.train_type:
                    threshold = 0.1
                elif "T0" in self.train_type:
                    threshold = 0.
                elif "T1" in self.train_type:
                    threshold = 0.01
                else:
                    threshold = 0.05

                if "bin" in self.train_type:
                    method = "bin"
                elif "batch" in self.train_type:
                    method = "batch"
                else:
                    method = "opt"

                if "N1000" in self.train_type:
                    n_samples = 1000
                else:
                    n_samples = 200

                est_stepsize = False
                if "ada" in self.train_type:
                    est_stepsize = True
                keep_d = min(50, int(np.prod(X.shape[1:])))
                self.eps_x = get_pca_ellipsoid_proj_fn_list(X, y, self.norm, keep_d=keep_d, threshold=threshold,
                        n_samples=n_samples, method=method, est_stepsize=est_stepsize, n_jobs=12, random_state=self.random_state,
                        cache_filename=cache_filename, info_cache_filename=idx_cache_filename,
                        is_img_data=is_img_data)

            elif "ellip" in self.train_type:
                if "ellip5" in self.train_type:
                    lamb = 5
                elif "ellip3" in self.train_type:
                    lamb = 3
                else:
                    lamb = 1
                self.eps_x = get_ellipsoid_proj_fn_list(X, y, self.norm, lamb=lamb,
                        n_jobs=4, random_state=self.random_state, cache_filename=cache_filename)
            else:
                raise ValueError(f"Unsupported train type: {self.train_type}.")
        else:
            if self.train_type is not None:
                raise ValueError(f"Unlabeled data for training type {self.train_type} not supported.")
            unlabeled_ds = self._get_dataset(unX, sample_weights=sample_weights, training=True,
                                             radius=self.eps_x, is_img_data=is_img_data)

        dataset = self._get_dataset(X, y, sample_weights=sample_weights, training=True,
                                    radius=self.eps_x, is_img_data=is_img_data)
        return self.fit_dataset(dataset, unlabeled_ds=unlabeled_ds, verbose=verbose,
                                is_img_data=is_img_data, normalized=normalized)

    def fit_dataset(self, dataset, unlabeled_ds=None, verbose=None, is_img_data=True, normalized=False):
        if verbose is None:
            verbose = 0 if not DEBUG else 1
        log_interval = 1

        history = []
        loss_fn = get_loss(self.loss_name, reduction="none")
        scheduler = get_scheduler(self.optimizer, n_epochs=self.epochs, loss_name=self.loss_name)

        if 'cat' in self.loss_name or "cusgrad" in self.loss_name:
            self.current_eps = torch.zeros((len(dataset), 1)).to(self.device)
            self.current_eps.requires_grad_(False)

        train_loader = torch.utils.data.DataLoader(dataset,
            batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        if unlabeled_ds is not None:
            assert NotImplementedError("not supporting unlabeled data")

        test_loader = None
        if self.tst_ds is not None:
            if isinstance(self.tst_ds, VisionDataset):
                ts_dataset = self.tst_ds
            else:
                tstX, tsty = self.tst_ds
                ts_dataset = self._get_dataset(tstX, tsty, training=False, is_img_data=is_img_data)
            test_loader = torch.utils.data.DataLoader(ts_dataset,
                batch_size=32, shuffle=False, num_workers=self.num_workers)

        for epoch in range(self.start_epoch, self.epochs+1):
            train_loss, train_acc = 0., 0.
            if self.current_eps is not None:
                print(self.current_eps.mean(), self.current_eps.min(), self.current_eps.max())

            for data in tqdm(train_loader, desc=f"Epoch {epoch}"):
                self.model.train()

                idx, x, y, w, eps_x = (d.to(self.device) for d in data)
                glob_eps = self.eps
                if self.train_type is None:
                    eps_x = self.eps
                elif "1nnregion" in self.train_type or "ellip" in self.train_type or 'subvor' in self.train_type:
                    if hasattr(self.eps_x[0], "stepsize"):
                        glob_eps = torch.from_numpy(
                            np.array([self.eps_x[i].stepsize for i in eps_x])).float().to(self.device)
                    eps_x = [self.eps_x[i] for i in eps_x]
                    

                clip_img = False
                if len(x.shape) == 4 and is_img_data:
                    clip_img = True

                current_eps = self.current_eps[idx] if self.current_eps is not None else None
                params = {
                    'sample_weight': w,
                    'n_classes': self.n_classes,
                    'norm': self.norm,
                    'device': self.device,
                    'eps': eps_x,
                    'clip_img': clip_img,
                    'normalized': normalized,
                    'loss_name': self.loss_name,
                    #'reduction': 'mean',
                    'glob_eps': glob_eps,
                    'current_epoch': epoch,
                    'total_epochs': self.epochs,
                    'current_eps': current_eps,
                }
                self.optimizer.zero_grad()

                if 'cat' in self.loss_name or "cusgrad" in self.loss_name:
                    outputs, loss, updated_current_eps = get_outputs_loss(
                        self.model, self.optimizer, loss_fn, x, y, **params
                    )
                    self.current_eps[idx] = updated_current_eps.clone().detach()
                else:
                    outputs, loss = get_outputs_loss(
                        self.model, self.optimizer, loss_fn, x, y, **params
                    )

                loss = w * loss

                loss = loss.mean()
                loss.backward()
                self.optimizer.step()

                if (epoch - 1) % log_interval == 0:
                    train_loss += loss.item() * len(x)
                    train_acc += (outputs.argmax(dim=1)==y).sum().float().item()

                    self.model.eval()
                    if self.eval_callbacks is not None:
                        for cb_fn in self.eval_callbacks:
                            cb_fn(self.model, train_loader, self.device)

            current_lr = self.optimizer.state_dict()['param_groups'][0]['lr']
            scheduler.step()
            self.start_epoch = epoch

            if (epoch - 1) % log_interval == 0:
                print(f"current LR: {current_lr}")
                self.model.eval()
                history.append({
                    'epoch': epoch,
                    'lr': current_lr,
                    'trn_loss': train_loss / len(train_loader.dataset),
                    'trn_acc': train_acc / len(train_loader.dataset),
                    'current_eps': self.current_eps.cpu().numpy if self.current_eps is not None else None,
                })
                print('epoch: {}/{}, train loss: {:.3f}, train acc: {:.3f}'.format(
                    epoch, self.epochs, history[-1]['trn_loss'], history[-1]['trn_acc']))

                if self.tst_ds is not None:
                    tst_loss, tst_acc = self._calc_eval(test_loader, loss_fn)
                    history[-1]['tst_loss'] = tst_loss
                    history[-1]['tst_acc'] = tst_acc
                    print('             test loss: {:.3f}, test acc: {:.3f}'.format(
                          history[-1]['tst_loss'], history[-1]['tst_acc']))

        if test_loader is not None:
            del test_loader
        del train_loader
        gc.collect()

        return history
