import pytorch_lightning as pl


class EEGCodec(pl.LightningModule):
    def __init__(self) -> None:
        super().__init__()
