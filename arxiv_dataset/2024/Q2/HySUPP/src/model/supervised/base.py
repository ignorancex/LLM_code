"""
Base model declaration for supervised unmixing methods
"""

from src.model.base import UnmixingModel


class SupervisedUnmixingModel(UnmixingModel):
    def __init__(
        self,
    ):
        super().__init__()

    def compute_abundances(
        self,
        Y,
        E,
        *args,
        **kwargs,
    ):
        raise NotImplementedError(f"Solver is not implemented for {self}")
