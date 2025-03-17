import time
import torch
import torch.nn.functional as F

from argparse import ArgumentParser
from loggers.exp_logger import ExperimentLogger
from datasets.exemplars_dataset import ExemplarsDataset


class Inc_Learning_Appr:
    """Basic class for implementing incremental learning approaches"""

    def __init__(self, model, device, nepochs=100, lr=0.05, lr_min=1e-4, lr_factor=3, lr_patience=5, clipgrad=10000,
                 momentum=0, wd=0, multi_softmax=False, wu_nepochs=0, wu_lr_factor=1, fix_bn=False,
                 eval_on_train=False, nc_first=None, num_tasks=10, template="",
                 logger: ExperimentLogger = None, exemplars_dataset: ExemplarsDataset = None):
        self.model = model
        self.device = device
        self.nepochs = nepochs
        self.lr = lr
        self.lr_min = lr_min
        self.lr_factor = lr_factor
        self.lr_patience = lr_patience
        self.clipgrad = clipgrad
        self.momentum = momentum
        self.wd = wd
        self.multi_softmax = multi_softmax
        self.logger = logger
        self.exemplars_dataset = exemplars_dataset
        self.warmup_epochs = wu_nepochs
        self.warmup_lr = lr * wu_lr_factor
        self.warmup_loss = torch.nn.CrossEntropyLoss()
        self.fix_bn = fix_bn
        self.eval_on_train = eval_on_train
        self.optimizer = None
        self.scheduler = None
        self.nc_first = nc_first
        self.num_tasks = num_tasks
        self.template = template

    @staticmethod
    def extra_parser(args):
        """Returns a parser containing the approach specific parameters"""
        parser = ArgumentParser()
        return parser.parse_known_args(args)

    @staticmethod
    def exemplars_dataset_class():
        """Returns a exemplar dataset to use during the training if the approach needs it
        :return: ExemplarDataset class or None
        """
        return None

    def _get_optimizer(self, t):
        """Returns the optimizer"""
        return torch.optim.SGD(self.model.parameters(), lr=self.lr, weight_decay=self.wd, momentum=self.momentum)

    def _init_optimizer(self, t):
        pass

    def check_trainable_params(self):
        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {enabled}")

    def train(self, t, trn_loader, val_loader, num_old, num_cur):
        """Main train structure"""
        self.pre_train_process(t, trn_loader)
        self.train_loop(t, trn_loader, val_loader, num_old, num_cur)
        self.post_train_process(t, trn_loader)

    def pre_train_process(self, t, trn_loader):
        """Runs before training all epochs of the task (before the train session)"""
        pass

    def train_loop(self, t, trn_loader, val_loader, num_old, num_cur):
        """Contains the epochs loop"""
        # add exemplars to train_loader
        if len(self.exemplars_dataset) > 0 and t > 0:
            trn_loader = torch.utils.data.DataLoader(trn_loader.dataset + self.exemplars_dataset,
                                                     batch_size=trn_loader.batch_size,
                                                     shuffle=True,
                                                     num_workers=trn_loader.num_workers,
                                                     pin_memory=trn_loader.pin_memory)

        self._init_optimizer(t)
        # TRAINING -- contains the epochs loop
        for e in range(self.nepochs):
            # Train
            train_loss, train_time = self.train_epoch(t, trn_loader, num_cur)
            self.scheduler.step()
            print('| Epoch {:3d}, time={:5.1f}s | lr: {:.5f}, loss={:.5f}'.format(e + 1, train_time,
                                                                                  self.optimizer.param_groups[0]['lr'],
                                                                                  train_loss / len(trn_loader)), end='')
            print()
        # EXEMPLAR MANAGEMENT -- select training subset
        self.exemplars_dataset.collect_exemplars(self.model, trn_loader, val_loader.dataset.transform)

    def post_train_process(self, t, trn_loader):
        """Runs after training all the epochs of the task (after the train session)"""
        pass

    def train_epoch(self, t, trn_loader, num_cur):
        """Runs a single epoch"""
        clock0 = time.time()
        self.model.train()
        losses = 0.
        for images, captions, targets in trn_loader:
            # Forward current model
            output = self.forward(images.to(self.device), captions, task_id=t, train=True, trained_classes=num_cur)
            loss = self.criterion(t, output, targets.to(self.device))
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clipgrad)
            self.optimizer.step()
            losses += loss.item()
        clock1 = time.time()
        return losses, clock1 - clock0

    def forward(self, images, captions, task_id=0, train=False, trained_classes=0):
        return self.model(images)

    def eval(self, t, val_loader, trained_classes, last, cur):
        """Contains the evaluation code"""
        with torch.no_grad():
            total_loss, total_acc_taw, total_acc_tag, total_num = 0, 0, 0, 0
            self.model.eval()
            for images, captions, targets in val_loader:
                # Forward current model
                outputs = self.forward(images.to(self.device), captions, t, False, trained_classes)
                loss = self.criterion(t, outputs, targets.to(self.device))
                hits_taw, hits_tag = self.calculate_metrics(outputs, targets, last, cur)
                # Log
                total_loss += loss.item() * len(targets)
                total_acc_taw += hits_taw.sum().item()
                total_acc_tag += hits_tag.sum().item()
                total_num += len(targets)
        return total_loss / total_num, total_acc_taw / total_num, total_acc_tag / total_num

    def calculate_metrics(self, logits, targets, last, cur):
        """Contains the main Task-Aware and Task-Agnostic metrics"""
        tag_probs = logits.softmax(dim=-1).cpu()
        tag_pred = tag_probs.argmax(1)
        hits_tag = (tag_pred.to(self.device) == targets.to(self.device)).float()

        taw_logits = logits[:, last:cur]
        taw_probs = taw_logits.softmax(dim=-1).cpu()
        taw_top_labels = taw_probs.argmax(1)
        taw_top_labels += last
        hits_taw = (taw_top_labels.to(self.device) == targets.to(self.device)).float()
        return hits_taw, hits_tag

    def criterion(self, t, outputs, targets):
        """Returns the loss value"""
        return F.cross_entropy(outputs, targets.to(self.device))
