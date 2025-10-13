import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback


class HPScheduler(Callback):
    def __init__(self,epochs, alpha_ms, alpha_hs, alpha_fusion) -> None:
        super(HPScheduler, self).__init__()
        self.epochs = epochs
        self.alpha_ms = alpha_ms
        self.alpha_hs = alpha_hs
        self.alpha_fusion = alpha_fusion

    def on_validation_epoch_end(self, trainer: pl.Trainer,
                                      pl_module: pl.LightningModule):

        epoch = trainer.current_epoch
        if epoch in self.epochs:
            i = self.epochs.index(epoch)
            pl_module.loss_criterion.alpha_ms = self.alpha_ms[i]
            pl_module.loss_criterion.alpha_hs = self.alpha_hs[i]
            pl_module.loss_criterion.alpha_fusion = self.alpha_fusion[i]

