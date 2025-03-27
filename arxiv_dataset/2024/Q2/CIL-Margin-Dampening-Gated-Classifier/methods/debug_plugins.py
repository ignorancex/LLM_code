import os
import pickle
from collections import defaultdict
from copy import deepcopy

import torch
from avalanche.core import SupervisedPlugin, Template, CallbackResult
from avalanche.models import avalanche_forward
from avalanche.training.plugins import ReplayPlugin
from avalanche.training.templates import SupervisedTemplate
from torch.utils.data import DataLoader


class LogitsDebugPlugin(SupervisedPlugin, supports_distributed=True):
    def __init__(
            self,
            saving_path,
            mem_size: int = 200,
            classes=10,
            batch_size=None,
            batch_size_mem=None,
            task_balanced_dataloader=False,
            storage_policy=None,
    ):
        super().__init__()

        self.replay_buffer = None
        self.history = defaultdict(list)
        self.mem_size = mem_size
        self.batch_size = batch_size
        self.batch_size_mem = batch_size_mem
        self.task_balanced_dataloader = task_balanced_dataloader

        self.saving_path = os.path.join(saving_path, 'logits.pkl')
        self.eval_saving_path = os.path.join(saving_path, 'eval_scores.pkl')
        os.makedirs(saving_path, exist_ok=True)

    def after_eval_exp(self, strategy: Template, *args,
                       **kwargs) -> CallbackResult:
        a = 0
        if len(strategy.evaluator.all_metric_results) > 0:
            with open(self.eval_saving_path, 'wb') as f:
                pickle.dump(strategy.evaluator.all_metric_results, f)

    def before_training_exp(self, strategy: "SupervisedTemplate", **kwargs):
        if strategy.experience.current_experience == 0:
            return

        for p in strategy.plugins:
            if isinstance(p, ReplayPlugin):
                self.replay_buffer = p.storage_policy.buffer

                self.update_logits(strategy, self.replay_buffer)

    def after_training_exp(self, strategy: Template, *args, **kwargs):
        with open(self.saving_path, 'wb') as f:
            pickle.dump(self.history, f)

    def after_training_epoch(
            self, strategy: Template, *args, **kwargs
    ) -> CallbackResult:
        if strategy.experience.current_experience == 0:
            return

        self.update_logits(strategy, self.replay_buffer)

    @torch.no_grad()
    def update_logits(self, strategy, dataset):
        tid = strategy.experience.current_experience
        dataloader = DataLoader(dataset,
                                batch_size=32)

        all_logits = []
        all_probs = []
        all_labels = []

        for x, y, t in dataloader:
            x = x.to(strategy.device)

            pred = strategy.model(x, t)
            all_logits.append(pred)

            all_probs.append(torch.softmax(pred, -1))
            all_labels.append(y)

        all_logits = torch.cat(all_logits, 0).cpu().numpy()
        all_probs = torch.cat(all_probs, 0).cpu().numpy()
        all_labels = torch.cat(all_labels, 0).cpu().numpy()

        self.history[tid].append((all_logits, all_probs, all_labels))


class GradientsDebugPlugin(SupervisedPlugin, supports_distributed=True):
    def __init__(
            self,
            saving_path,
            mem_size: int = 200,
            classes=10,
            batch_size=None,
            batch_size_mem=None,
            task_balanced_dataloader=False,
            storage_policy=None,
    ):
        super().__init__()

        self.replay_buffer = None
        self.history = defaultdict(list)
        self.mem_size = mem_size
        self.batch_size = batch_size
        self.batch_size_mem = batch_size_mem
        self.task_balanced_dataloader = task_balanced_dataloader

        self.saving_path = os.path.join(saving_path, 'gradients.pkl')
        self.history = []
        os.makedirs(saving_path, exist_ok=True)

    def before_training_exp(self, strategy: "SupervisedTemplate", **kwargs):
        if strategy.experience.current_experience == 0:
            return

        self.past_model = deepcopy(strategy.model)

        if hasattr(strategy, 'storage_policy'):
            self.replay_buffer = strategy.storage_policy.buffer
        else:
            for p in strategy.plugins:
                if hasattr(p, 'storage_policy'):
                    self.replay_buffer = p.storage_policy.buffer

    def after_training_exp(self, strategy: Template, *args, **kwargs):
        with open(self.saving_path, 'wb') as f:
            pickle.dump(self.history, f)

    def before_training_epoch(
            self, strategy: Template, *args, **kwargs
    ) -> CallbackResult:
        if strategy.experience.current_experience == 0:
            return

        tid = strategy.experience.current_experience
        dataloader = DataLoader(self.replay_buffer,
                                batch_size=len(self.replay_buffer))

        for x, y, t, pl in dataloader:
            x = x.to(strategy.device)
            y = y.to(strategy.device)
            pl = pl.to(strategy.device)

            l = strategy.model(x)

            strategy.model.zero_grad()
            ce = torch.nn.functional.cross_entropy(l, y)
            ce.backward(retain_graph=True)
            ce_grads = {n: p.grad for n, p in strategy.model.named_parameters()
                        if
                        p.grad is not None}

            strategy.model.zero_grad()
            ce = torch.nn.functional.mse_loss(l, pl)
            ce.backward(retain_graph=True)

            mse_grads = {n: p.grad for n, p in strategy.model.named_parameters()
                         if
                         p.grad is not None}

            self.history.append((ce_grads, mse_grads))

        strategy.model.zero_grad()


class TrainDebugPlugin(SupervisedPlugin):
    def __init__(
            self,
            saving_path,
            mem_size: int = 200,
            classes=10,
            batch_size=None,
            batch_size_mem=None,
            task_balanced_dataloader=False,
            storage_policy=None,
    ):

        super().__init__()

        self.replay_buffer = None

        self.history = defaultdict(list)
        self.after_task_history = dict()

        self.mem_size = mem_size
        self.batch_size = batch_size
        self.batch_size_mem = batch_size_mem
        self.task_balanced_dataloader = task_balanced_dataloader

        self.saving_path = os.path.join(saving_path, 'logits.pkl')
        self.saving_path2 = os.path.join(saving_path, 'after_logits.pkl')

        self.eval_saving_path = os.path.join(saving_path, 'eval_scores.pkl')
        os.makedirs(saving_path, exist_ok=True)

        self.all_dataset = {}

    def after_training_exp(self, strategy: Template, *args, **kwargs):
        tid = strategy.experience.current_experience
        res = self.update_logits(strategy, self.replay_buffer)

        self.history[tid].append(res)
        self.after_task_history[tid] = res

        with open(self.saving_path, 'wb') as f:
            pickle.dump(self.history, f)

        with open(self.saving_path2, 'wb') as f:
            pickle.dump(self.after_task_history, f)

    # def before_training_exp(self, strategy: Template, *args, **kwargs):
    #     dataset = strategy.experience.dataset
    #     self.all_dataset.append(dataset)

    def after_eval_dataset_adaptation(self, strategy: Template, *args, **kwargs) -> CallbackResult:
        tid = strategy.experience.current_experience
        if tid not in self.all_dataset:
            self.all_dataset[tid] = strategy.experience.dataset
        # if tid >= len(self.all_dataset):
        #     self.all_dataset.append(strategy.experience.dataset)

    def before_training_epoch(
        self, strategy: Template, *args, **kwargs
    ) -> CallbackResult:
        if len(self.all_dataset) == 0:
            return

        tid = strategy.experience.current_experience
        res = self.update_logits(strategy, self.replay_buffer)
        self.history[tid].append(res)

    # def after_train_dataset_adaptation(
    #     self, strategy: Template, *args, **kwargs
    # ) -> CallbackResult:
    #
    #     dataset = strategy.experience.current_experience
    #     self.all_dataset.append(dataset)
    #
    #     tid = strategy.experience.current_experience
    #     res = self.update_logits(strategy, self.replay_buffer)
    #     self.after_task_history[tid] = res
    #
    #     with open(self.saving_path, 'wb') as f:
    #         pickle.dump(self.history, f)
    #
    #     with open(self.saving_path2, 'wb') as f:
    #         pickle.dump(self.after_task_history, f)

    @torch.no_grad()
    def update_logits(self, strategy, dataset):
        strategy.model.eval()

        res = {}
        for i, d in self.all_dataset.items():
            dataloader = DataLoader(d, batch_size=256, shuffle=False)

            all_logits = []
            all_probs = []
            all_labels = []

            for x, y, t in dataloader:
                x = x.to(strategy.device)

                pred = avalanche_forward(strategy.model, x, t)
                # strategy.model(x, t)

                if isinstance(pred, tuple):
                    pred, _ = pred
                    pred = torch.cat(pred, -1)

                all_logits.append(pred)

                all_probs.append(torch.softmax(pred, -1))
                all_labels.append(y)

            all_logits = torch.cat(all_logits, 0).cpu().numpy()
            all_probs = torch.cat(all_probs, 0).cpu().numpy()
            all_labels = torch.cat(all_labels, 0).cpu().numpy()

            res[i] = (all_logits, all_probs, all_labels)

        strategy.model.train()
        return res
