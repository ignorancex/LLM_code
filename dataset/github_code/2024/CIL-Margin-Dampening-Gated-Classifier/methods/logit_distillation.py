from copy import deepcopy
from typing import Optional

from avalanche.core import SupervisedPlugin, Template, CallbackResult
from avalanche.training import Replay, ClassBalancedBuffer
from avalanche.training.plugins import ReplayPlugin
from avalanche.training.plugins.evaluation import default_evaluator
from torch import nn
from torch.utils.data import DataLoader
import torch


class LogitDistillationPlugin(SupervisedPlugin, supports_distributed=True):
    def __init__(
            self,
            mem_size: int = 200,
            alpha=1,
            storage_policy: Optional["ExemplarsBuffer"] = None,
    ):
        super().__init__()

        self.alpha = alpha
        self.past_model = None
        if storage_policy is not None:  # Use other storage policy
            self.storage_policy = storage_policy
            assert storage_policy.max_size == mem_size
        else:  # Default
            self.storage_policy = ClassBalancedBuffer(max_size=mem_size,
                                                      adaptive_size=True)

    def after_training_exp(self, strategy: "SupervisedTemplate", **kwargs):
        self.storage_policy.update(strategy, **kwargs)

    def before_training_exp(self, strategy: Template, *args, **kwargs):
        self.past_model = deepcopy(strategy.model)
        self.past_model.eval()

    def before_backward(self, strategy: Template, *args,
                        **kwargs) -> CallbackResult:
        x = strategy.mb_x

        dataset = self.storage_policy.buffer
        if len(dataset) == 0:
            return

        px, _, pt = next(iter(DataLoader(dataset, len(x), True)))
        px = px.to(x.device)

        with torch.no_grad():
            pl = self.past_model(px, task_labels=pt)

        cl = strategy.model(px, task_labels=pt)
        past_reg_loss = nn.functional.kl_div(
            torch.log_softmax(cl, -1),
            torch.softmax(pl, -1), reduction='batchmean')

        strategy.loss += past_reg_loss * self.alpha


class LogitDistillation(Replay):
    """Experience replay strategy.

    See ReplayPlugin for more details.
    This strategy does not use task identities.
    """

    def __init__(
            self,
            model,
            optimizer,
            criterion,
            alpha=1,
            mem_size: int = 200,
            train_mb_size: int = 1,
            train_epochs: int = 1,
            eval_mb_size=None,
            device="cpu",
            plugins=None,
            evaluator=default_evaluator,
            eval_every=-1,
            **base_kwargs
    ):
        rp = LogitDistillationPlugin(mem_size, alpha)

        if plugins is None:
            plugins = [rp]
        else:
            plugins.append(rp)

        super().__init__(
            model,
            optimizer,
            criterion,
            train_mb_size=train_mb_size,
            train_epochs=train_epochs,
            eval_mb_size=eval_mb_size,
            device=device,
            plugins=plugins,
            evaluator=evaluator,
            eval_every=eval_every,
            **base_kwargs
        )


class ReplayDebug(Replay):
    """Experience replay strategy.

    See ReplayPlugin for more details.
    This strategy does not use task identities.
    """

    def __init__(
            self,
            model,
            optimizer,
            criterion,
            mem_size: int = 200,
            train_mb_size: int = 1,
            train_epochs: int = 1,
            eval_mb_size=None,
            device="cpu",
            plugins=None,
            evaluator=default_evaluator,
            eval_every=-1,
            **base_kwargs
    ):
        rp = ReplayPlugin(mem_size)
        if plugins is None:
            plugins = [rp]
        else:
            plugins.append(rp)
        super().__init__(
            model,
            optimizer,
            criterion,
            train_mb_size=train_mb_size,
            train_epochs=train_epochs,
            eval_mb_size=eval_mb_size,
            device=device,
            plugins=plugins,
            evaluator=evaluator,
            eval_every=eval_every,
            **base_kwargs
        )

        if plugins is None:
            plugins = [rp]
        else:
            plugins.append(rp)
        super().__init__(
            model,
            optimizer,
            criterion,
            train_mb_size=train_mb_size,
            train_epochs=train_epochs,
            eval_mb_size=eval_mb_size,
            device=device,
            plugins=plugins,
            evaluator=evaluator,
            eval_every=eval_every,
            **base_kwargs
        )
