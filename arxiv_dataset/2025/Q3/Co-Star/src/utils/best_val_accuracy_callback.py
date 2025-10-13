import pytorch_lightning as pl

class BestValAccuracyCallback(pl.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.best_val_accuracy = 0.0
        self.best_epoch = 0

    def on_validation_end(self, trainer, pl_module):
        val_accuracy = trainer.callback_metrics.get("val_accuracy")
        if val_accuracy is not None and val_accuracy > self.best_val_accuracy:
            self.best_val_accuracy = val_accuracy
            self.best_epoch = trainer.current_epoch
           # self.save_best_model(trainer, pl_module)

            # Print the best validation accuracy and epoch to the console
            print(f"New best validation accuracy: {self.best_val_accuracy:.4f} at epoch {self.best_epoch}")

            # Log best accuracy and epoch directly to the logger (e.g., W&B)
            if trainer.logger is not None:
                trainer.logger.log_metrics({
                    'best_val_accuracy': self.best_val_accuracy,
                    'best_val_epoch': self.best_epoch
                }, step=trainer.current_epoch)

    # def save_best_model(self, trainer, pl_module):
    #     save_path = f"{trainer.logger.save_dir}/{trainer.logger.name}/version_{trainer.logger.version}/best_model.ckpt"
    #     trainer.save_checkpoint(save_path)

    def on_train_end(self, trainer, pl_module):
        print(f"Training completed. Best Validation Accuracy: {self.best_val_accuracy:.4f} at epoch {self.best_epoch}")
