import abc

import torch


class AbstractTester(abc.ABC):
    """Abstract class for trainer."""

    @abc.abstractmethod
    def run_test(
        self,
        model,
        loader,
        device: torch.device,
        beam_size: int = 5,
        lm_beta: float = 0.4,
        ins_gamma: float = 1.2,
    ):
        pass
