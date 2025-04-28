from collections import namedtuple

import pytorch_lightning as pl


class EEGDataset(pl.LightningDataModule):
    def __init__(self) -> None:
        super().__init__()


EEGBatch = namedtuple(
    "EEGBatch",
    ("data", "id", "sample_id", "patient", "dataset"),
)
