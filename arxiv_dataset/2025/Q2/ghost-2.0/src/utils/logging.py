import os
import numpy as np
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from .preprocess import make_X_dict

class PeriodicCheckpoint(ModelCheckpoint):
    def __init__(self, every: int, dir: str):
        super().__init__()
        self.every = every
        self.dir = dir
        os.makedirs(self.dir, exist_ok=True) 

    def on_train_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, *args, **kwargs
    ):
        if trainer.current_epoch % self.every == 0:
            current = f"{self.dir}/epoch-{trainer.current_epoch}.ckpt"
            trainer.save_checkpoint(current)
    
        current = f"{self.dir}/last.ckpt"
        trainer.save_checkpoint(current)


class LogPredictionSamplesCallback(pl.Callback):
    def __init__(self, logger, n=4, log_train_freq = 1000):
        super(LogPredictionSamplesCallback, self).__init__()
        self.logger = logger
        self.n = n
        self.log_train_freq = log_train_freq
    
    def plot_images(self, pl_module, outputs, batch, mode=None): 
  
        X_dict = make_X_dict(
                    batch['face_arc'].detach(),
                    batch['face_wide'].detach(),
                    batch['face_wide_mask'].detach() 
                )

        if mode is None:
            columns = ['source', 'target', 'fake', 'mask'] 
            data = [
                list(map(lambda x: ((x.cpu().detach().numpy() + 1) / 2 * 255).astype(np.uint8), (x, y, z, k)))
    
                for x, y, z, k in zip(
                    X_dict['source']['face_wide'][:self.n, 0],
                    X_dict['target']['face_wide'][:self.n],
                    outputs['fake_rgbs'][:self.n],
                    (outputs['fake_segm'][:self.n].expand_as(outputs['fake_rgbs'][:self.n]))
                )
            ]
            data = np.nan_to_num(np.concatenate([np.concatenate(row, axis=2)[np.newaxis, ...] for row in data], axis=2))
            
        return data[0]
        
    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):                    
        if batch_idx == 0:
            data = self.plot_images(pl_module, outputs, batch)
            self.logger.experiment.add_image(f"Val samples {'self' if dataloader_idx == 0 else 'cross'}", data[::-1, :, :], trainer.global_step)
            
    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx):     
        if batch_idx % self.log_train_freq == 0:
            data = self.plot_images(pl_module, outputs, batch)
            self.logger.experiment.add_image(f"Train samples", data[::-1, :, :], trainer.global_step)

