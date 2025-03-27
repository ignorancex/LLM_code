from typing import Optional

import numpy as np

from dataset.augmentors import BaseAugmentor


class DualAugmentor(BaseAugmentor):
    """
    A class that applies 2 augmentors in parallel and creates a batch for SimCLR-based experiments.
    """

    def __init__(self, augmentor1: Optional[BaseAugmentor], augmentor2: Optional[BaseAugmentor]):
        assert issubclass(type(augmentor1), BaseAugmentor) or augmentor1 is None
        assert issubclass(type(augmentor2), BaseAugmentor) or augmentor2 is None

        self._augmentor1: BaseAugmentor = augmentor1
        self._augmentor2: BaseAugmentor = augmentor2

    def augment_single(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Dual augmentor overwrites augment_batch, use that instead")

    def augment_batch(self, batch: np.ndarray) -> np.ndarray:
        # We directly overwrite augment_batch instead of augment_single because of the nature of the augmentor.

        # NOTE - Important to copy, otherwise it's the same batch for both augmentation channels
        batch1: np.ndarray = batch.copy()
        batch2: np.ndarray = batch.copy()

        if self._augmentor1 is not None:
            batch1 = self._augmentor1.augment_batch(batch1)
        if self._augmentor2 is not None:
            batch2 = self._augmentor2.augment_batch(batch2)

        return np.vstack((batch1, batch2))
