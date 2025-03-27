import torch
import torch.nn.functional as F

from clip import tokenize
from .incremental_learning import Inc_Learning_Appr
from datasets.exemplars_dataset import ExemplarsDataset


class Appr(Inc_Learning_Appr):
    """
    Class implementing the VPT(Visual Prompt Tuning, ECCV 2022) approach
    described in https://arxiv.org/pdf/2204.04799.pdf
    Original code available at https://github.com/KMnP/vpt
    """

    def __init__(self, model, device, nepochs=100, lr=0.05, lr_min=1e-4, lr_factor=3, lr_patience=5, clipgrad=10000,
                 momentum=0, wd=0, multi_softmax=False, wu_nepochs=0, wu_lr_factor=1, fix_bn=False, eval_on_train=False,
                 nc_first=None, num_tasks=10, template="", logger=None, exemplars_dataset=None):
        super(Appr, self).__init__(model, device, nepochs, lr, lr_min, lr_factor, lr_patience, clipgrad, momentum, wd,
                                   multi_softmax, wu_nepochs, wu_lr_factor, fix_bn, eval_on_train, nc_first, num_tasks,
                                   template, logger, exemplars_dataset)
        self.total_classes = 0

    @staticmethod
    def exemplars_dataset_class():
        return ExemplarsDataset

    def _get_optimizer(self, t):
        """Returns the optimizer"""
        for name, parameter in self.model.named_parameters():
            if 'prompt_embeddings' not in name:
                parameter.requires_grad = False
        params = [self.model.prompt_embeddings] + \
                 [self.model.deep_prompt_embeddings]
        # double check
        self.check_trainable_params()
        return torch.optim.SGD(params, lr=self.lr, weight_decay=self.wd, momentum=self.momentum)

    def _init_optimizer(self, t=0):
        self.optimizer = self._get_optimizer(t)
        self.scheduler = torch.optim.lr_scheduler.ConstantLR(self.optimizer, factor=1.0)

    def train_loop(self, t, trn_loader, val_loader, num_old, num_cur):
        """Contains the epochs loop"""
        self.total_classes = num_cur
        super().train_loop(t, trn_loader, val_loader, num_old, num_cur)

    def forward(self, images, captions, task_id=0, train=False, trained_classes=0):
            logits = self.model.forward(images.to(self.device))
            return logits[:, :self.total_classes]

    def criterion(self, t, logits, targets, num_old=0, num_cur=0):
        # following the original paper, we mask out classes of non-current tasks when exemplars is unavailable
        if len(self.exemplars_dataset) == 0:
            negative_inf_tensor = torch.full_like(logits, float('-inf'))
            logits[:, :num_old] = negative_inf_tensor[:, :num_old]
        loss = F.cross_entropy(logits, targets.to(self.device))
        return loss

