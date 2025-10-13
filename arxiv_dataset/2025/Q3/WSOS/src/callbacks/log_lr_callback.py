from lightning import Callback


class LogLRCallback(Callback):
    def __init__(self):
        super().__init__()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # Get the current learning rate from the optimizer
        current_lr = trainer.optimizers[0].param_groups[0]['lr']
        pl_module.log({'learning_rate': current_lr, 'global_step': trainer.global_step})
