import time
import torch
import torch.nn.functional as F

from .incremental_learning import Inc_Learning_Appr
from datasets.exemplars_dataset import ExemplarsDataset


class Appr(Inc_Learning_Appr):
    """
    Class implementing the CoOp(Learning to Prompt for Vision-Language Models, IJCV 2022) approach
    described in https://arxiv.org/pdf/2109.01134.pdf
    Original code available at https://github.com/KaiyangZhou/CoOp
    """

    def __init__(self, model, device, nepochs=100, lr=0.05, lr_min=1e-4, lr_factor=3, lr_patience=5, clipgrad=10000,
                 momentum=0, wd=0, multi_softmax=False, wu_nepochs=0, wu_lr_factor=1, fix_bn=False, eval_on_train=False,
                 nc_first=None, num_tasks=10, template="", logger=None, exemplars_dataset=None):
        super(Appr, self).__init__(model, device, nepochs, lr, lr_min, lr_factor, lr_patience, clipgrad, momentum, wd,
                                   multi_softmax, wu_nepochs, wu_lr_factor, fix_bn, eval_on_train, nc_first, num_tasks,
                                   template, logger, exemplars_dataset)

    @staticmethod
    def exemplars_dataset_class():
        return ExemplarsDataset

    def _get_optimizer(self, t):
        """Returns the optimizer"""
        for name, parameter in self.model.named_parameters():
            if 'prompt_learner' not in name:
                parameter.requires_grad = False
        params = list(self.model.prompt_learner.parameters())
        # double check
        self.check_trainable_params()
        return torch.optim.SGD(params, lr=self.lr, weight_decay=self.wd, momentum=self.momentum)

    def _init_optimizer(self, t=0):
        self.optimizer = self._get_optimizer(t)
        self.scheduler = torch.optim.lr_scheduler.ConstantLR(self.optimizer, factor=1.0)

    def train_loop(self, t, trn_loader, val_loader, num_old, num_cur):
        """Contains the epochs loop"""
        super().train_loop(t, trn_loader, val_loader, num_old, num_cur)

    def train_epoch(self, t, trn_loader, num_cur=0):
        clock0 = time.time()
        self.model.train()
        losses = 0.
        for images, captions, targets in trn_loader:
            # Forward current model
            output = self.model(images.to(self.device), num_cur)
            loss = F.cross_entropy(output, targets.to(self.device))
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clipgrad)
            self.optimizer.step()
            losses += loss.item()
        clock1 = time.time()
        return losses, clock1 - clock0

    def forward(self, images, captions, task_id=0, train=False, trained_classes=0):
        return self.model(images, trained_classes)
