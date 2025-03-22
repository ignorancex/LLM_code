from copy import deepcopy

import numpy as np
import torch
from avalanche.training.plugins import EvaluationPlugin
from avalanche.training.plugins.evaluation import default_evaluator
from avalanche.training.templates import SupervisedTemplate
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader


class LogitsDataset:
    def __init__(self, base_datasets, logits, features=None, classes=None):
        self.base_dataset = base_datasets
        self.logits = logits
        self.features = features
        self.current_classes = classes

    def __len__(self):
        return len(self.logits)

    def __getitem__(self, item):
        if self.features is None:
            return *self.base_dataset[item], self.logits[item]

        return *self.base_dataset[item], self.logits[item], self.features[item]

    def train(self):
        self.base_dataset = self.base_dataset.train()
        return self

    def eval(self):
        self.base_dataset = self.base_dataset.eval()
        return self

    def subset(self, indexes):
        self.base_dataset = self.base_dataset.subset(indexes)
        self.logits = self.logits[indexes]

        if self.features is not None:
            self.features = self.features[indexes]

        return self


class MarginDampening(SupervisedTemplate):
    def __init__(self, model,
                 optimizer: Optimizer,
                 mem_size: int,
                 train_mb_size: int = 1,
                 train_epochs: int = 1,
                 alpha: float = 0,
                 past_task_reg=1,
                 regularize_logits=False,
                 rehearsal_metric='kl',
                 margin_type='adaptive',
                 margin=0.5,
                 warm_up_epochs=-1,
                 future_task_reg=0,
                 future_margin=3,
                 gamma=1,
                 use_logits_memory=False,
                 cumulative_memory=True,
                 reg_sampling_bs=None,
                 eval_mb_size: int = None,
                 device=None,
                 plugins=None,
                 batch_size_mem=None,
                 criterion=None,
                 evaluator: EvaluationPlugin = default_evaluator,
                 eval_every=-1,
                 ):

        assert margin_type in ['fixed', 'adaptive', 'normal', 'mean',
                               'max_mean']
        assert 0 <= margin <= 1

        assert rehearsal_metric in ['kl', 'mse']

        self.future_margins = []
        self.tasks_nclasses = dict()
        self.current_mb_size = 0
        self.regularize_logits = regularize_logits
        self.future_logits = None

        self.cumulative_memory = cumulative_memory

        self.double_sampling = -1 if reg_sampling_bs is None else reg_sampling_bs

        self.past_task_reg = past_task_reg
        self.margin = margin
        self.margin_type = margin_type
        self.warm_up_epochs = warm_up_epochs

        self.future_margin = future_margin
        self.future_task_reg = future_task_reg

        self.alpha = alpha
        # Past task logits weight
        self.gamma = gamma

        self.logit_regularization = rehearsal_metric
        self.tau = 1

        self.memory_size = mem_size

        self.use_logits_memory = use_logits_memory

        if batch_size_mem is None:
            self.batch_size_mem = train_mb_size
        else:
            self.batch_size_mem = batch_size_mem

        self.is_finetuning = False
        self.mb_future_features = None

        self.past_dataset = {}
        self.past_dataset_tasks = {}
        self.past_model = None

        super().__init__(
            model=model, optimizer=optimizer, criterion=None,
            train_mb_size=train_mb_size,
            train_epochs=train_epochs,
            eval_mb_size=eval_mb_size, device=device,
            plugins=plugins,
            evaluator=evaluator, eval_every=eval_every)

    def _after_forward(self, **kwargs):
        self.mb_output, self.future_logits = self.mb_output
        super()._after_forward(**kwargs)

    def _after_eval_forward(self, **kwargs):
        self.mb_output, self.future_logits = self.mb_output
        self.mb_output = torch.cat(self.mb_output, -1)
        super()._after_eval_forward(**kwargs)

    def _after_training_iteration(self, **kwargs):
        super()._after_training_iteration(**kwargs)

    def _after_training_exp(self, **kwargs):
        if self.use_logits_memory:
            self._after_training_exp_logits()
        else:
            self._after_training_exp_f(**kwargs)

        super()._after_training_exp(**kwargs)

        classes_so_far = len(self.experience.classes_seen_so_far)

    def sample_past_batch(self, batch_size, strict=True):
        classes = self.past_dataset
        ln = len(classes)

        if ln <= 0:
            return None

        samples_per_task = max(batch_size // ln, 1)
        rest = batch_size % ln

        if rest > 0:
            to_add = np.random.choice(list(classes))
        else:
            to_add = -1

        x, y, t = [], [], []
        ls = []

        for c in classes:
            d = self.past_dataset[c]

            if c == to_add:
                bs = samples_per_task + rest
            else:
                bs = samples_per_task

            batch = next(iter(DataLoader(d,
                                         batch_size=bs,
                                         shuffle=True)))

            if len(batch) == 3:
                ot_x, ot_y, ot_tid = batch
            else:
                ot_x, ot_y, ot_tid, l = batch
                ls.append(l)

            ot_tid = torch.full_like(ot_tid, self.past_dataset_tasks[c])
            x.append(ot_x)
            y.append(ot_y)
            t.append(ot_tid)

        if len(ls) > 0:
            ls = torch.cat(ls)
        return torch.cat(x, 0), torch.cat(y, 0), torch.cat(t, 0), ls

    def _before_forward_f(self, **kwargs):

        super()._before_forward(**kwargs)

        if len(self.past_dataset) == 0:
            return None

        batch = self.sample_past_batch(len(self.mb_x))
        if len(batch) == 3:
            x, y, t = batch
        else:
            x, y, t, l = batch
            self.past_logits = l

        self.current_mb_size = len(self.mb_x)

        current_task = self.experience.current_experience
        self.mbatch[2] = torch.full_like(self.mbatch[2], current_task)

        self.mbatch[0] = torch.cat((self.mbatch[0], x.to(self.device)))
        self.mbatch[1] = torch.cat((self.mbatch[1], y.to(self.device)))
        self.mbatch[2] = torch.cat((self.mbatch[2], t.to(self.device)))

    @torch.no_grad()
    def _after_training_exp_f(self, **kwargs):

        if self.cumulative_memory:
            samples_to_save = self.memory_size // len(
                self.experience.classes_seen_so_far)
        else:
            samples_to_save = self.memory_size

        tid = self.experience.current_experience

        for k, d in self.past_dataset.items():
            indexes = np.arange(len(d))

            if len(indexes) > samples_to_save:
                selected = np.random.choice(indexes, samples_to_save, False)
                d = d.train().subset(selected)
                self.past_dataset[k] = d

        dataset = self.experience.dataset
        ys = np.asarray(dataset.targets)

        for y in np.unique(ys):
            indexes = np.argwhere(ys == y).reshape(-1)
            indexes = np.random.choice(indexes, samples_to_save, False)

            self.past_dataset[y] = dataset.train().subset(indexes)
            self.past_dataset_tasks[y] = tid

        # if self.experience.current_experience > 0:
        self.past_model = deepcopy(self.model)
        self.past_model.eval()

    @torch.no_grad()
    def _after_training_exp_logits(self, **kwargs):
        tid = self.experience.current_experience

        samples_to_save = self.memory_size // len(
            self.experience.classes_seen_so_far)

        for k, d in self.past_dataset.items():
            indexes = np.arange(len(d))

            if len(indexes) > samples_to_save:
                selected = np.random.choice(indexes, samples_to_save, False)
                d = d.train().subset(selected)

                self.past_dataset[k] = d

        dataset = self.experience.dataset
        ys = np.asarray(dataset.targets)

        self.model.eval()
        for y in np.unique(ys):
            indexes = np.argwhere(ys == y).reshape(-1)
            indexes = np.random.choice(indexes, samples_to_save, False)
            all_logits = []
            for x, _, _ in DataLoader(dataset.eval().subset(indexes),
                                      batch_size=self.eval_mb_size):
                x = x.to(self.device)

                logits = self.model(x)[0]
                logits = torch.cat(logits, -1)
                all_logits.append(logits.cpu())

            all_logits = torch.cat(all_logits, 0)

            d = LogitsDataset(dataset.train().subset(indexes), all_logits)

            self.past_dataset[y] = d
            self.past_dataset_tasks[y] = tid

        self.past_model = deepcopy(self.model)
        self.past_model.eval()

    def _before_forward(self, **kwargs):
        super()._before_forward(**kwargs)
        self._before_forward_f(**kwargs)

    @torch.no_grad()
    def _before_training_exp(self, **kwargs):
        tid = self.experience.current_experience

        super()._before_training_exp(**kwargs)

        if (len(self.past_dataset) > 0 and not self.use_logits_memory):
            self.past_model = deepcopy(self.model)
            self.past_model.eval()

        self.future_margins.append(self.future_margin)

        if self.use_logits_memory:
            self.model.eval()

            for k, d in self.past_dataset.items():
                all_logits = []
                for x, y, _, l in DataLoader(d.eval(),
                                             batch_size=self.train_mb_size):
                    _l = self.model(x.to(self.device))[0]
                    _l = _l[tid].cpu()
                    l = torch.cat((l, _l), -1)
                    all_logits.append(l)

                all_logits = torch.cat(all_logits, 0)
                d.logits = all_logits

                self.past_dataset[k] = d

            self.model.train()

    def _before_training_epoch(self, **kwargs):
        super()._before_training_epoch(**kwargs)

        tid = self.experience.current_experience
        if tid == 0:
            return

        with torch.no_grad(), torch.inference_mode():
            logits = []

            for x, y, t in self.dataloader:
                x, y = x.to(self.device), y.to(self.device)

                preds, _ = self.model(x)
                past = torch.cat(preds, -1)

                preds = torch.cat(preds, -1)
                preds = torch.softmax(preds, -1)
                cl = preds[range(len(preds)), y]

                logits.append(cl)

            logits = torch.cat(logits, 0)

            if self.margin_type == 'mean':
                self._margin = 1 - logits.mean()
            elif self.margin_type != 'normal':
                self.margin_distribution = torch.distributions.normal.Normal(
                    logits.mean(), logits.std())

    def criterion(self):
        tid = self.experience.current_experience

        if not self.is_training:
            loss_val = nn.functional.cross_entropy(self.mb_output,
                                                   self.mb_y)

            return loss_val

        if len(self.past_dataset) == 0:
            pred = torch.cat(self.mb_output, -1)
            if self.future_logits is not None:
                pred = torch.cat((pred, self.future_logits), -1)

            loss = nn.functional.cross_entropy(pred, self.mb_y,
                                               label_smoothing=0)
        else:
            ce_loss = 0
            margin_loss = 0
            ce_den = 0

            for t in torch.unique(self.mb_task_id):
                if t < tid and self.alpha <= 0:
                    continue

                mask = self.mb_task_id == t
                ce_den += mask.sum().item()
                ce_m = None

                y = self.mb_y[mask]
                co = torch.cat(self.mb_output[t.item():], -1)[mask]

                if t > 0:
                    past = torch.cat(self.mb_output[:t.item()], -1)[mask]
                    distr = torch.cat((past, co), -1)

                    if self.regularize_logits:
                        mx_current_classes = distr[range(len(co)), y]
                        past_max = distr[:, :past.shape[-1]].max(-1).values
                    else:

                        distr = torch.softmax(distr, -1)

                        mx_current_classes = distr[range(len(co)), y]
                        past_max = distr[:, :past.shape[-1]].max(-1).values

                    m = nn.functional.one_hot(y, distr.shape[-1])
                    nm = 1 - m
                    pm = (distr * nm).max(-1).values
                    ce_m = pm > mx_current_classes

                    y = y - past.shape[-1]

                    if self.margin_type == 'fixed':
                        margin = self.margin
                    elif self.margin_type == 'normal':
                        margin = self.margin_distribution.rsample([len(distr)])
                        margin = torch.clip(margin, 0, 1)
                        margin = margin * self.margin
                        margin = torch.minimum(margin, torch.full_like(margin, (
                                    1 / (distr.shape[-1] - 1))))
                    elif self.margin_type == 'mean':
                        margin = (1 - mx_current_classes.mean()) * self.margin
                    elif self.margin_type == 'max_mean':
                        margin = past_max.mean() * self.margin
                    else:
                        margin = (1 / (distr.shape[-1] - 1))

                    margin_dist = torch.relu(
                        past_max - mx_current_classes + margin)

                    margin_loss = margin_dist.mean()

                if self.future_logits is not None:
                    co = torch.cat((co, self.future_logits[mask]), -1)

                loss = nn.functional.cross_entropy(co, y, reduction='sum')

                if t < tid:
                    loss = loss * self.alpha

                ce_loss += loss

            w = 1
            if self.warm_up_epochs > 0:
                w = min(1, (
                            self.clock.train_exp_epochs / self.warm_up_epochs) ** 2)

            margin_loss = margin_loss * self.past_task_reg * w
            ce_loss = (ce_loss / ce_den)

            loss = ce_loss + margin_loss

            if self.gamma > 0:
                factor = 1

                if len(self.past_dataset) > 0:
                    x, y, tids = self.mbatch
                    curr_logits = self.mb_output
                    mask = tids != tid

                    if self.use_logits_memory:
                        past_logits = self.past_logits.to(x.device)
                    else:
                        with torch.no_grad():
                            past_logits, _ = self.past_model(x)
                            past_logits = torch.cat(past_logits, -1)[mask]

                    if self.future_task_reg > 0:
                        co = torch.cat(curr_logits[:-1], -1)
                        past_logits = past_logits[:, :co.shape[-1]]
                    else:
                        co = torch.cat(curr_logits, -1)

                    co = co[mask]

                    if self.logit_regularization == 'mse':
                        past_reg_loss = nn.functional.mse_loss(co, past_logits,
                                                               reduction='mean')
                    else:
                        past_reg_loss = nn.functional.kl_div(
                            torch.log_softmax(co, -1),
                            torch.softmax(past_logits, -1),
                            reduction='batchmean')

                    loss = loss + past_reg_loss * self.gamma * factor

            future_reg = 0
            if self.future_task_reg > 0:
                x, y, tids = self.mbatch

                mask = tids != tid
                cc = self.mb_output[tid].shape[-1]

                co = torch.cat(self.mb_output, -1)[mask]

                distr = torch.softmax(co, -1)
                p, c = distr[:, :-cc], distr[:, -cc:]

                margin = (1 / (distr.shape[-1] - 1))

                future_mx = c.max(-1).values
                current_mx = p.max(-1).values

                margin_dist = torch.relu(future_mx - current_mx + margin)

                future_reg += margin_dist.mean()

                future_reg = future_reg * self.future_task_reg

                loss += future_reg

        return loss
