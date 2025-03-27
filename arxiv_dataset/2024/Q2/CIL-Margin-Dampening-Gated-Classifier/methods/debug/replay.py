from avalanche.training import Replay
from avalanche.training.plugins import ReplayPlugin
from avalanche.training.plugins.evaluation import default_evaluator
from torch.nn import Module


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
        eval_mb_size = None,
        device  = "cpu",
        plugins = None,
        evaluator = default_evaluator,
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
