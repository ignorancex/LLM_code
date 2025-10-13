from lightning import Callback


class LearningRateLoggerCallback(Callback):
    def on_epoch_end(self, trainer, pl_module):
        current_lr = pl_module.trainer.optimizers[0].param_groups[0]['lr']
        trainer.logger.experiment.log({"lr": current_lr}, step=trainer.global_step)