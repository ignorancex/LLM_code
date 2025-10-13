import torch
import pytorch_lightning as pl
import torchmetrics as metrics

from .models import RITS
from .utils import _to_var


class LitRITS(pl.LightningModule):
    def __init__(
            self,
            rnn_hid_size: int = 100,
            impute_weight: float = 0.3,
            label_weight: float = 0.7,
            data_dim: int = 6,
            seq_len: int = 12,
            class_level: str = 'L5',
            lr: float = 1e-3,
            wd: float = 0,
            optimizer='adam',
            scheduler='cosine'
    ):
        super(LitRITS, self).__init__()
        self.model = RITS(
            rnn_hid_size=rnn_hid_size,
            impute_weight=impute_weight,
            label_weight=label_weight,
            data_dim=data_dim,
            seq_len=seq_len,
            class_level=class_level
        )

        self.lr = lr
        self.wd = wd
        self.optimizer = optimizer
        self.scheduler = scheduler

        # save hyper-parameters to self.hparams (auto-logged by W&B)
        self.save_hyperparameters()

        # Prepare metrics
        self.train_acc = metrics.Accuracy(num_classes=self.model.num_classes)
        self.train_aucroc = metrics.AUROC(num_classes=self.model.num_classes, average='macro')
        self.train_f1 = metrics.F1Score(num_classes=self.model.num_classes, average='macro')

        self.val_acc = metrics.Accuracy(num_classes=self.model.num_classes)
        self.val_aucroc = metrics.AUROC(num_classes=self.model.num_classes, average='macro')
        self.val_f1 = metrics.F1Score(num_classes=self.model.num_classes, average='macro')

        self.test_acc = metrics.Accuracy(num_classes=self.model.num_classes)
        self.test_aucroc = metrics.AUROC(num_classes=self.model.num_classes, average='macro')
        self.test_f1 = metrics.F1Score(num_classes=self.model.num_classes, average='macro')

    def forward(self, x):
        return self.model(x, direct='forward')

    def training_step(self, batch, batch_idx):
        data = _to_var(batch)
        ret = self(data)

        self.train_acc(ret['predictions'], ret['labels'])
        self.train_aucroc(ret['predictions'], ret['labels'])
        self.train_f1(ret['predictions'], ret['labels'])

        self.log('train/loss', ret['loss'], prog_bar=True)
        self.log('train/acc', self.train_acc, on_step=True, on_epoch=True)
        self.log('train/auc_roc', self.train_aucroc, on_step=True, on_epoch=True)
        self.log('train/f1_macro', self.train_f1, on_step=True, on_epoch=True)

        return ret['loss']

    def validation_step(self, batch, batch_idx):
        data = _to_var(batch)
        ret = self(data)

        self.val_acc(ret['predictions'], ret['labels'])
        self.val_aucroc(ret['predictions'], ret['labels'])
        self.val_f1(ret['predictions'], ret['labels'])

        self.log('val/loss', ret['loss'], prog_bar=True)
        self.log('val/acc', self.val_acc, on_step=True, on_epoch=True)
        self.log('val/auc_roc', self.val_aucroc, on_step=True, on_epoch=True)
        self.log('val/f1_macro', self.val_f1, on_step=True, on_epoch=True)

        return ret['loss']

    def test_step(self, batch, batch_idx):
        data = _to_var(batch)
        ret = self(data)

        self.test_acc(ret['predictions'], ret['labels'])
        self.test_aucroc(ret['predictions'], ret['labels'])
        self.test_f1(ret['predictions'], ret['labels'])

        self.log('test/loss', ret['loss'], prog_bar=True)
        self.log('test/acc', self.test_acc, on_step=False, on_epoch=True)
        self.log('test/auc_roc', self.test_aucroc, on_step=False, on_epoch=True)
        self.log('test/f1_macro', self.test_f1, on_step=False, on_epoch=True)

        return ret['loss']

    def configure_optimizers(self):
        """ Define optimizers and LR schedulers """
        if self.optimizer == 'adam':
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.wd)
        elif self.optimizer == 'sgd':
            optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, weight_decay=self.wd)

        if self.scheduler == 'cosine':
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.trainer.max_epochs, 0)
        if self.scheduler == 'step':
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5)

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': lr_scheduler,
                'interval': 'epoch',
                'frequency': 1
            }
        }
