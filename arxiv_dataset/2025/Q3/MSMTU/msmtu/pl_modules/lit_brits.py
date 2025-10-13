import torch
import pytorch_lightning as pl
from torchmetrics import Accuracy, F1Score, ConfusionMatrix, PearsonCorrCoef
import numpy as np
from msmtu.metrics import MSE, MAE

from .models import BRITS
from .utils import _to_var


class LitBRITS(pl.LightningModule):
    def __init__(
            self,
            idx_to_class,
            rnn_hid_size: int = 100,
            impute_weight: float = 0.25,
            label_weight: float = 0.75,
            data_dim: int = 6,
            seq_len: int = 12,
            ancillary_dim: int = 0,
            embedding_dim: int = 50,
            lr: float = 1e-3,
            wd: float = 0,
            optimizer: str = 'adam',
            scheduler: str = 'cosine',
            pre_trained_path: str = None,
    ):
        super(LitBRITS, self).__init__()

        self.model = BRITS(
            rnn_hid_size=rnn_hid_size,
            impute_weight=impute_weight,
            label_weight=label_weight,
            data_dim=data_dim,
            seq_len=seq_len,
            output_dim=len(idx_to_class),
            ancillary_dim=ancillary_dim,
            embedding_dim=embedding_dim
        )

        # save hyper-parameters to self.hparams (auto-logged by W&B)
        self.save_hyperparameters()

        self._create_metrics(self.model.output_dim)

    def _create_metrics(self, num_classes):
        # Prepare metrics
        self.train_acc = Accuracy(task='multiclass', num_classes=num_classes)
        self.train_f1_class = F1Score(task='multiclass', num_classes=num_classes, average=None)

        self.val_acc = Accuracy(task='multiclass', num_classes=num_classes)
        self.val_f1_class = F1Score(task='multiclass', num_classes=num_classes, average=None)

        self.test_acc = Accuracy(task='multiclass', num_classes=num_classes)
        self.test_f1_class = F1Score(task='multiclass', num_classes=num_classes, average=None)

        self.train_rmse_class = MSE(squared=False, num_classes=num_classes, averaged=False)
        self.val_rmse_class = MSE(squared=False, num_classes=num_classes, averaged=False)
        self.test_rmse_class = MSE(squared=False, num_classes=num_classes, averaged=False)

        self.train_mae_class = MAE(absolute=True, num_classes=num_classes, averaged=False)
        self.val_mae_class = MAE(absolute=True, num_classes=num_classes, averaged=False)
        self.test_mae_class = MAE(absolute=True, num_classes=num_classes, averaged=False)

        # Correlation coefficient metrics with "num_outputs" set to "num_classes"
        self.val_pearson = PearsonCorrCoef(num_outputs=num_classes)
        self.test_pearson = PearsonCorrCoef(num_outputs=num_classes)

    def forward(self, x, stage='train'):
        return self.model(x, stage)

    def predict_step(self, batch, batch_idx, dataloader_idx: int = 0):
        data = _to_var(batch)
        ret = self(data)

        return ret

    def training_step(self, batch, batch_idx):
        data = _to_var(batch)
        ret = self(data, stage='train')

        # Classification metrics
        self.train_acc(ret['predictions'], ret['labels'])
        batch_f1 = self.train_f1_class(ret['predictions'], ret['labels']).mean()

        self.log('train/loss', ret['loss'], prog_bar=True)
        self.log('train/acc', self.train_acc, on_step=True, on_epoch=True, prog_bar=True)

        # Regression metrics
        batch_rmse = self.train_rmse_class(ret['predictions'], ret['probs']).mean()
        batch_mae = self.train_mae_class(ret['predictions'], ret['probs']).mean()

        self.log('train/f1-score_step', batch_f1, prog_bar=True)
        self.log('train/rmse_step', batch_rmse, prog_bar=True)
        self.log('train/mae_step', batch_mae, prog_bar=True)

        return ret['loss']

    def training_epoch_end(self, outputs):

        metrics = {
            'f1': self.train_f1_class,
            'rmse': self.train_rmse_class,
            'mae': self.train_mae_class
        }

        self._log_class_metrics(stage='train', metrics=metrics)

    def validation_step(self, batch, batch_idx):
        data = _to_var(batch)
        ret = self(data, stage='test')

        # Classification metrics
        self.val_acc(ret['predictions'], ret['labels'])
        self.val_f1_class(ret['predictions'], ret['labels'])

        self.log('val/loss', ret['loss'], on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/acc', self.val_acc, on_step=False, on_epoch=True)

        # Regression metrics
        self.val_rmse_class(ret['predictions'], ret['probs'])
        self.val_mae_class(ret['predictions'], ret['probs'])
        self.val_pearson(ret['predictions'], ret['probs'])

        return ret['loss']

    def validation_epoch_end(self, outputs):
        metrics = {
            'f1': self.val_f1_class,
            'rmse': self.val_rmse_class,
            'mae': self.val_mae_class,
            'pearson': self.val_pearson
        }

        self._log_class_metrics(stage='val', metrics=metrics)

    def test_step(self, batch, batch_idx):
        data = _to_var(batch)
        ret = self(data, stage='test')

        self.test_acc(ret['predictions'], ret['labels'])
        self.test_f1_class(ret['predictions'], ret['labels'])

        self.log('test/loss', ret['loss'])
        self.log('test/acc', self.test_acc, on_step=False, on_epoch=True)

        # Regression metrics
        self.test_rmse_class(ret['predictions'], ret['probs'])
        self.test_mae_class(ret['predictions'], ret['probs'])
        self.test_pearson(ret['predictions'], ret['probs'])

        return ret['loss']

    def test_epoch_end(self, outputs):
        # Compute metrics
        metrics = {
            'f1': self.test_f1_class,
            'rmse': self.test_rmse_class,
            'mae': self.test_mae_class,
            'pearson': self.test_pearson
        }

        self._log_class_metrics(stage='test', metrics=metrics)

    def configure_optimizers(self):
        """ Define optimizers and LR schedulers """
        if self.hparams.optimizer == 'adam':
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.wd)
        elif self.hparams.optimizer == 'sgd':
            optimizer = torch.optim.SGD(self.model.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.wd)

        if self.hparams.scheduler == 'cosine':
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.trainer.max_epochs, 0)
        if self.hparams.scheduler == 'step':
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5)

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': lr_scheduler,
                'interval': 'epoch',
                'frequency': 1
            }
        }

    def change_last_layer(self, idx_to_class: dict, class_weight=None):
        """ Change the last layer of the model to match the new number of classes """
        self.hparams.idx_to_class = idx_to_class
        self.hparams.class_weight = class_weight
        new_num_classes = len(self.hparams.idx_to_class)
        self.model.change_last_layer(new_num_classes, class_weight)
        self._create_metrics(new_num_classes)

    def _log_class_metrics(self, stage: str, metrics: dict):
        """ Log class-wise metrics for a given stage """
        for metric, object in metrics.items():
            computed_metric = object.compute()
            self.log(f'{stage}/{metric}_epoch', computed_metric.mean(), prog_bar=True)  # Log mean metric

            # Log class-wise metric
            for idx, class_name in self.hparams.idx_to_class.items():
                self.log(f'{stage}/{metric}_{class_name}_epoch', computed_metric[idx])

            # Reset metric
            object.reset()
