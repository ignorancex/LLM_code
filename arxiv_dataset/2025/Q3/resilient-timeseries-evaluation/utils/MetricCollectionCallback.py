from pytorch_lightning.callbacks import Callback

class MetricCollectionCallback(Callback):
    """
    Callback to collect the train and validation loss during training.
    """
    
    def __init__(self):
        """
        Initialize the MetricCollectionCallback.
        """
        super().__init__()
        self.metrics = {"train_loss": [], "val_loss": []}

    def on_train_epoch_end(self, trainer, pl_module):
        """
        Called on train epoch end. Logs the losses.

        Parameters
        ----------
        trainer : pl.Trainer
            The PyTorch Lightning that this callback is attached to.
        pl_module : pl.Module
            The PyTorch Lightning module that is attached to the trainer.å
        """
        # Collect train and validation loss
        train_loss = trainer.callback_metrics.get("train_loss")
        val_loss = trainer.callback_metrics.get("val_loss")
        if train_loss is not None:
            self.metrics["train_loss"].append(train_loss.cpu().detach().item())
        if val_loss is not None:
            self.metrics["val_loss"].append(val_loss.cpu().detach().item())