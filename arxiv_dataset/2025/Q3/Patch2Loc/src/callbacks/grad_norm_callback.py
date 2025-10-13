from lightning import Callback

from src.utils.utils import gradient_norm


class GradNormCallback(Callback):
    """
    Logs the gradient norm.
    """

    def __init__(self):
        self.optimizer_indices = {}
    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        if optimizer not in self.optimizer_indices:
            self.optimizer_indices[optimizer] = len(self.optimizer_indices)
        optimizer_idx = self.optimizer_indices[optimizer]
        pl_module.log(f'grad_norm_{optimizer_idx}', gradient_norm(optimizer))
