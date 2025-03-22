import time
import torch
import torch.nn.functional as F

from argparse import ArgumentParser
from .incremental_learning import Inc_Learning_Appr
from datasets.exemplars_dataset import ExemplarsDataset


class Appr(Inc_Learning_Appr):
    """
    Class implementing the DualPrompt(DualPrompt: Complementary Prompting for Rehearsal-free Continual Learning, ECCV 2022) approach
    described in https://arxiv.org/pdf/2204.04799.pdf
    Original code available at https://github.com/google-research/l2p
    PyTorch version implemented code can be found at https://github.com/JH-LEE-KR/dualprompt-pytorch
    """

    def __init__(self, model, original_model, device, nepochs=100, lr=0.05, lr_min=1e-4, lr_factor=3, lr_patience=5,
                 clipgrad=10000, momentum=0, wd=0, multi_softmax=False, wu_nepochs=0, wu_lr_factor=1, fix_bn=False,
                 eval_on_train=False, nc_first=None, num_tasks=10, template="", logger=None, exemplars_dataset=None,
                 pull_constraint=True, pull_constraint_coef=0.5):
        super(Appr, self).__init__(model, device, nepochs, lr, lr_min, lr_factor, lr_patience, clipgrad, momentum, wd,
                                   multi_softmax, wu_nepochs, wu_lr_factor, fix_bn, eval_on_train, nc_first, num_tasks,
                                   template, logger, exemplars_dataset)
        self.total_classes = 0
        self.original_model = original_model
        self.pull_constraint = pull_constraint
        self.pull_constraint_coef = pull_constraint_coef

    @staticmethod
    def exemplars_dataset_class():
        return ExemplarsDataset

    @staticmethod
    def extra_parser(args):
        """Returns a parser containing the approach specific parameters"""
        parser = ArgumentParser()
        parser.add_argument('--pull_constraint', action='store_false', required=False,
                            help='optimize cosine similarity between key and query (default=%(default)s)')
        parser.add_argument('--pull_constraint_coef', default=1.0, type=float, required=False,
                            help='trade-off between cross-entropy and cosine similarity (default=%(default)s)')
        return parser.parse_known_args(args)

    def _get_optimizer(self, t):
        """Returns the optimizer"""
        for name, parameter in self.model.named_parameters():
            if 'prompt' not in name and 'head' not in name:
                parameter.requires_grad = False
        params = list(self.model.e_prompt.parameters()) + \
                 list(self.model.head.parameters()) + \
                 [self.model.g_prompt]
        # double check
        self.check_trainable_params()
        # return torch.optim.Adam(params, lr=self.lr, weight_decay=self.wd)
        return torch.optim.SGD(params, lr=self.lr, momentum=self.momentum, weight_decay=self.wd)

    def pre_train_process(self, t, trn_loader):
        """Runs before training all epochs of the task (before the train session)"""
        self.freeze(self.original_model)

    def _init_optimizer(self, t=0):
        self.optimizer = self._get_optimizer(t)
        self.scheduler = torch.optim.lr_scheduler.ConstantLR(self.optimizer, factor=1.0)

    def freeze(self, model):
        for p in model.parameters():
            p.requires_grad = False

    def train_loop(self, t, trn_loader, val_loader, num_old, num_cur):
        """Contains the epochs loop"""
        # add exemplars to train_loader
        if len(self.exemplars_dataset) > 0 and t > 0:
            trn_loader = torch.utils.data.DataLoader(trn_loader.dataset + self.exemplars_dataset,
                                                     batch_size=trn_loader.batch_size,
                                                     shuffle=True,
                                                     num_workers=trn_loader.num_workers,
                                                     pin_memory=trn_loader.pin_memory)

        # TRAINING -- contains the epochs loop
        self.total_classes = num_cur

        # transfer learned prompts
        self.transfer_prompt(t)

        self._init_optimizer(t)

        for e in range(self.nepochs):
            # Train
            train_loss, train_time = self.train_epoch(t, trn_loader, num_old, num_cur)
            self.scheduler.step()
            print('| Epoch {:3d}, time={:5.1f}s | lr: {:.5f}, loss={:.5f}'.format(e + 1, train_time,
                                                                                  self.optimizer.param_groups[0]['lr'],
                                                                                  train_loss / len(trn_loader)), end='')
            print()
        # EXEMPLAR MANAGEMENT -- select training subset
        self.exemplars_dataset.collect_exemplars(self.model, trn_loader, val_loader.dataset.transform)

    def forward(self, images, captions, task_id=0, train=False, trained_classes=0):
        with torch.no_grad():
            images = images.type(self.original_model.dtype)
            if self.original_model is not None:
                cls_features = self.original_model(images.to(self.device))
            else:
                cls_features = None
        output = self.model(images.to(self.device), task_id=task_id, cls_features=cls_features, train=train)
        return output

    def train_epoch(self, t, trn_loader, num_old=0, num_cur=0):
        clock0 = time.time()
        self.model.train()
        losses = 0.
        for images, captions, targets in trn_loader:
            # Forward current model
            images = images.type(self.original_model.dtype)
            with torch.no_grad():
                if self.original_model is not None:
                    cls_features = self.original_model(images.to(self.device))
                else:
                    cls_features = None
            outputs = self.model(images.to(self.device), task_id=t, cls_features=cls_features, train=True)
            loss = self.criterion(t, outputs, targets, num_old, num_cur)
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clipgrad)
            self.optimizer.step()
            losses += loss.item()
        clock1 = time.time()
        return losses, clock1 - clock0

    def criterion(self, t, outputs, targets, num_old=0, num_cur=0):
        logits = outputs['logits']
        # following the original paper, we mask out classes of non-current tasks when exemplars is unavailable
        if len(self.exemplars_dataset) == 0:
            negative_inf_tensor = torch.full_like(logits, float('-inf'))
            logits[:, :num_old] = negative_inf_tensor[:, :num_old]
            logits[:, num_cur:] = negative_inf_tensor[:, num_cur:]
        else:
            logits = logits[:, :self.total_classes]
        loss = F.cross_entropy(logits, targets.to(self.device))
        if self.pull_constraint and 'reduce_sim' in outputs:
            loss = loss - self.pull_constraint_coef * outputs['reduce_sim']
        return loss

    def calculate_metrics(self, outputs, targets, last, cur):
        logits = outputs['logits'][:, :self.total_classes]
        tag_probs = logits.softmax(dim=-1).cpu()
        tag_pred = tag_probs.argmax(1)
        hits_tag = (tag_pred.to(self.device) == targets.to(self.device)).float()

        taw_logits = logits[:, last:cur]
        taw_probs = taw_logits.softmax(dim=-1).cpu()
        taw_top_labels = taw_probs.argmax(1)
        taw_top_labels += last
        hits_taw = (taw_top_labels.to(self.device) == targets.to(self.device)).float()
        return hits_taw, hits_tag

    def transfer_prompt(self, t):
        # Transfer previous learned prompt params to the new prompt
        if self.model.config.prompt_pool and self.model.config.shared_prompt_pool:
            if t > 0:
                prev_start = (t - 1) * self.model.config.top_k
                prev_end = t * self.model.config.top_k

                cur_start = prev_end
                cur_end = (t + 1) * self.model.config.top_k

                if (prev_end > self.model.config.pool_size) or (cur_end > self.model.config.pool_size):
                    pass
                else:
                    cur_idx = (slice(None), slice(None),
                               slice(cur_start, cur_end)) if self.model.config.use_prefix_tune_for_e_prompt else (
                    slice(None), slice(cur_start, cur_end))
                    prev_idx = (slice(None), slice(None),
                                slice(prev_start, prev_end)) if self.model.config.use_prefix_tune_for_e_prompt else (
                    slice(None), slice(prev_start, prev_end))

                with torch.no_grad():
                    self.model.e_prompt.prompt.grad.zero_()
                    self.model.e_prompt.prompt[cur_idx] = self.model.e_prompt.prompt[prev_idx]
                    self.optimizer.param_groups[0]['params'] = self.model.parameters()
        # Transfer previous learned prompt param keys to the new prompt
        if self.model.config.prompt_pool and self.model.config.shared_prompt_key:
            if t > 0:
                prev_start = (t - 1) * self.model.config.top_k
                prev_end = t * self.model.config.top_k

                cur_start = prev_end
                cur_end = (t + 1) * self.model.config.top_k

                with torch.no_grad():
                    self.model.e_prompt.prompt_key.grad.zero_()
                    self.model.e_prompt.prompt_key[cur_idx] = self.model.e_prompt.prompt_key[prev_idx]
                    self.optimizer.param_groups[0]['params'] = self.model.parameters()
