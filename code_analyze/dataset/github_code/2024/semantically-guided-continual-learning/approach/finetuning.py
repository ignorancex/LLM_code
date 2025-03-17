import time
import torch
import torch.nn.functional as F
from argparse import ArgumentParser

from clip import tokenize
from .incremental_learning import Inc_Learning_Appr
from datasets.exemplars_dataset import ExemplarsDataset


class Appr(Inc_Learning_Appr):
    """
    Class implementing the finetuning baseline(contrastive loss for training)
    """

    def __init__(self, model, device, nepochs=100, lr=0.05, lr_min=1e-4, lr_factor=3, lr_patience=5, clipgrad=10000,
                 momentum=0, wd=0, multi_softmax=False, wu_nepochs=0, wu_lr_factor=1, fix_bn=False, eval_on_train=False,
                 nc_first=None, num_tasks=10, template="", logger=None, exemplars_dataset=None, all_outputs=False):
        super(Appr, self).__init__(model, device, nepochs, lr, lr_min, lr_factor, lr_patience, clipgrad, momentum, wd,
                                   multi_softmax, wu_nepochs, wu_lr_factor, fix_bn, eval_on_train, nc_first, num_tasks,
                                   template, logger, exemplars_dataset)
        self.all_out = all_outputs

    @staticmethod
    def exemplars_dataset_class():
        return ExemplarsDataset

    @staticmethod
    def extra_parser(args):
        """Returns a parser containing the approach specific parameters"""
        parser = ArgumentParser()
        parser.add_argument('--all-outputs', action='store_true', required=False,
                            help='Allow all weights related to all outputs to be modified (default=%(default)s)')
        return parser.parse_known_args(args)

    def _get_optimizer(self, t):
        """Returns the optimizer"""
        for name, parameter in self.model.named_parameters():
            if 'visual.transformer.resblocks.11' not in name:
                parameter.requires_grad = False
        params = list(self.model.model.visual.transformer.resblocks[11].parameters())
        # double check
        self.check_trainable_params()
        return torch.optim.SGD(params, lr=self.lr, weight_decay=self.wd, momentum=self.momentum)

    def _init_optimizer(self, t=0):
        self.optimizer = self._get_optimizer(t)
        self.scheduler = torch.optim.lr_scheduler.ConstantLR(self.optimizer, factor=1.0)

    def train_loop(self, t, trn_loader, val_loader, num_old, num_cur):
        """Contains the epochs loop"""
        self.model.build_classifier_from_text(self.template, self.device, num_old, num_cur)
        super().train_loop(t, trn_loader, val_loader, num_old, num_cur)

    def train_epoch(self, t, trn_loader, *args):
        """Runs a single epoch"""
        clock0 = time.time()
        self.model.train()
        losses = 0.
        for images, captions, targets in trn_loader:
            # Forward current model
            logits_per_image = self.forward(images, captions, t, train=True)
            loss = self.criterion(t, logits_per_image, targets)
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clipgrad)
            self.optimizer.step()
            losses += loss.item()
        clock1 = time.time()
        return losses, clock1 - clock0

    def forward(self, images, captions, task_id=0, train=False, trained_classes=0):
        image_features = self.model.encode_image(images.to(self.device))
        if train:  # contrastive loss
            text = tokenize(captions)
            text_features = self.model.encode_text(text.to(self.device))
            text_features = text_features / text_features.norm(dim=1, keepdim=True)
        else:  # using text features to get predictions
            text_features = torch.cat(self.model.text_features_per_task, dim=0)

        # normalized features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.model.model.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()
        return logits

    def criterion(self, t, logits_per_image, targets):
        if logits_per_image.shape[0] == logits_per_image.shape[1]:
            logits_per_text = logits_per_image.t()
            labels = torch.arange(len(logits_per_image)).to(self.device)
            image_loss = F.cross_entropy(logits_per_image, labels)
            text_loss = F.cross_entropy(logits_per_text, labels)
            loss = (image_loss + text_loss) / 2
        else:
            loss = F.cross_entropy(logits_per_image, targets.to(self.device))
        return loss
