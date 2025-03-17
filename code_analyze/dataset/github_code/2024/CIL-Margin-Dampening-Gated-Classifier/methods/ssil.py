from copy import deepcopy
from itertools import cycle
from typing import Optional, Union, List, Callable

import torch
from avalanche.core import SupervisedPlugin
from avalanche.models import avalanche_forward
from avalanche.training import ClassBalancedBuffer
from avalanche.training.plugins import EvaluationPlugin
from avalanche.training.plugins.evaluation import default_evaluator
from avalanche.training.templates import SupervisedTemplate
from torch.nn import CrossEntropyLoss
from torch.optim.optimizer import Optimizer


class SeparatedSoftmax(SupervisedTemplate):
    """
    Implements the SS-IL: Separated Softmax Strategy,
    from the "SS-IL: Separated Softmax for Incremental Learning"
    paper, Hongjoon Ahn et. al, https://arxiv.org/abs/2003.13947
    """

    def __init__(
            self,
            model: torch.nn.Module,
            optimizer: Optimizer,
            criterion=CrossEntropyLoss(reduction='sum'),
            mem_size: int = 200,
            tau: float = 2,
            batch_size_mem: Optional[int] = None,
            train_mb_size: int = 1,
            train_epochs: int = 1,
            eval_mb_size: Optional[int] = 1,
            device: Union[str, torch.device] = "cpu",
            plugins: Optional[List[SupervisedPlugin]] = None,
            evaluator: Union[
                EvaluationPlugin, Callable[[], EvaluationPlugin]
            ] = default_evaluator,
            eval_every=-1,
            peval_mode="epoch",
    ):
        """
        :param model: PyTorch model.
        :param optimizer: PyTorch optimizer.
        :param criterion: loss function.
        :param mem_size: int       : Fixed memory size
        :param tau: float          : The temperature used in the KL loss
        :param batch_size_mem: int : Size of the batch sampled from the buffer
        :param transforms: Callable: Transformations to use for
                                     both the dataset and the buffer data, on
                                     top of already existing
                                     test transformations.
                                     If any supplementary transformations
                                     are applied to the
                                     input data, it will be
                                     overwritten by this argument
        :param train_mb_size: mini-batch size for training.
        :param train_passes: number of training passes.
        :param eval_mb_size: mini-batch size for eval.
        :param device: PyTorch device where the model will be allocated.
        :param plugins: (optional) list of StrategyPlugins.
        :param evaluator: (optional) instance of EvaluationPlugin for logging
            and metric computations. None to remove logging.
        :param eval_every: the frequency of the calls to `eval` inside the
            training loop. -1 disables the evaluation. 0 means `eval` is called
            only at the end of the learning experience. Values >0 mean that
            `eval` is called every `eval_every` experiences and at the end of
            the learning experience.
        :param peval_mode: one of {'experience', 'iteration'}. Decides whether
            the periodic evaluation during training should execute every
            `eval_every` experience or iterations (Default='experience').
        """
        super().__init__(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            train_mb_size=train_mb_size,
            train_epochs=train_epochs,
            eval_mb_size=eval_mb_size,
            device=device,
            plugins=plugins,
            evaluator=evaluator,
            eval_every=eval_every,
            peval_mode=peval_mode,
        )

        self.T = tau
        self.seen_classes = []
        self.past_model = None
        if batch_size_mem is None:
            self.batch_size_mem = train_mb_size
        else:
            self.batch_size_mem = batch_size_mem
        self.mem_size = mem_size

        self.storage_policy = ClassBalancedBuffer(
            max_size=self.mem_size, adaptive_size=True
        )
        self.replay_loader = None

    def _before_training_exp(self, **kwargs):
        self.storage_policy.update(self, **kwargs)
        buffer = self.storage_policy.buffer
        if (
                len(buffer) >= self.batch_size_mem
                and self.experience.current_experience > 0
        ):
            self.replay_loader = cycle(
                torch.utils.data.DataLoader(
                    buffer,
                    batch_size=self.batch_size_mem,
                    shuffle=True,
                    drop_last=True,
                )
            )
        super()._before_training_exp(**kwargs)

    def _after_training_exp(self, **kwargs):
        self.storage_policy.update(self, **kwargs)
        self.seen_classes.append(self.experience.classes_in_this_experience)
        super()._after_training_exp(**kwargs)

    def _before_train_dataset_adaptation(self, **kwargs):

        self.past_model = deepcopy(self.model)
        self.past_model.eval()
        super()._before_train_dataset_adaptation(**kwargs)

    def training_epoch(self, **kwargs):
        classes = self.experience.classes_in_this_experience

        for self.mbatch in self.dataloader:
            if self._stop_training:
                break

            self._unpack_minibatch()
            self._before_training_iteration(**kwargs)

            if self.replay_loader is not None:
                self.mb_buffer_x, self.mb_buffer_y, self.mb_buffer_tid = next(
                    self.replay_loader
                )
                self.mb_buffer_x, self.mb_buffer_y, self.mb_buffer_tid = (
                    self.mb_buffer_x.to(self.device),
                    self.mb_buffer_y.to(self.device),
                    self.mb_buffer_tid.to(self.device),
                )

            self.optimizer.zero_grad()
            self.loss = self._make_empty_loss()

            # Forward
            self._before_forward(**kwargs)
            self.mb_output = self.forward()
            if self.replay_loader is not None:
                self.mb_buffer_out = avalanche_forward(
                    self.model, self.mb_buffer_x, self.mb_buffer_tid
                )
            self._after_forward(**kwargs)

            # Loss & Backward
            if self.replay_loader is None:
                self.loss = self.criterion()
            else:
                past_logits = self.past_model(self.mb_buffer_x,
                                              self.mb_buffer_y)

                curr_ce = self._criterion(self.mb_output[:, -len(classes):],
                                          self.mb_y - min(classes))
                prev_ce = self._criterion(self.mb_buffer_out, self.mb_buffer_y)

                ce = ((curr_ce + prev_ce) /
                      (len(past_logits) + len(self.mb_output)))

                kd = 0

                for t in range(len(self.seen_classes)):
                    s = min(self.seen_classes[t])
                    e = max(self.seen_classes[t]) + 1

                    soft_target = torch.softmax(past_logits[:, s:e] / self.T,
                                                dim=1)
                    output_log = torch.log_softmax(
                        self.mb_buffer_out[:, s:e] / self.T, dim=1)

                    kd += torch.nn.functional.kl_div(output_log, soft_target,
                                                     reduction='batchmean') * (
                                      self.T ** 2)

                self.loss = kd + ce

            self._before_backward(**kwargs)
            self.backward()
            self._after_backward(**kwargs)

            # Optimization step
            self._before_update(**kwargs)
            self.optimizer_step()
            self._after_update(**kwargs)

            self._after_training_iteration(**kwargs)
