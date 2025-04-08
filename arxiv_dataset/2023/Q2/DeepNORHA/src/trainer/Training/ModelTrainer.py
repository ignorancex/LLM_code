import pytorch_lightning as pl
import torch
import matplotlib.pyplot as plt
import random
import numpy as np
from sklearn.manifold import TSNE
import seaborn as sns

class ModelTrainer(pl.LightningModule):
    def __init__(self, model, optimizer, loss, dataModule, monitor, activate_loss=None, **config) -> None:
        super().__init__()
        self._model = model
        self._optimizer = optimizer
        self._loss = loss
        self._dataModule = dataModule
        self._monitor = monitor
        self._config = config
        self._activate_loss = activate_loss
        self.tensor_test = []
        self.label_test = []

    def to(self, device):
        super().to(device)


    def configure_optimizers(self):
        '''
        Configuration of the opimizer used in the model
        '''
        cls = getattr(torch.optim, self._optimizer['Type'])
        optimizer = cls(self._model.parameters(), **{k:v for k,v in self._optimizer.items() if k!='Type'})
	# optimizer.param_groups[0]['capturable'] = torch.Tensor([True])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                               mode='min',
                                                               factor=0.2,
                                                               patience=15,
                                                               min_lr=1e-6,
                                                               verbose=True)
        return {
        "optimizer": optimizer,
        "lr_scheduler": {
            "scheduler": scheduler,
            "monitor": 'mse_loss'
        },
    }
    def training_step(self, batch, batch_nb):
        '''
        training step inside the training loop
        it is made it for each batch of data'''


        x,y = batch
        x=x.to(torch.float)
        y_hat = self._model.forward(x)

        loss = self._loss(y_hat,x)
        #scores = torch.sum((y_hat - x) ** 2, dim=tuple(range(1, y_hat.dim())))
        #loss = torch.mean(scores)
        monitor = self._monitor.onTrainingStep(self, self._model, x, y, y_hat, loss, {})
        return monitor

    def training_epoch_end(self, outputs):
        '''
        Function called at the end of the training loop in one epoch
        outputs: values saved in the return of the training_step'''
        
        self.log('mse_loss',torch.stack([x['loss'] for x in outputs]).mean())
        self._monitor.onTrainingEpochEnd(self, self._model, outputs, {})

    def validation_step(self, batch, batch_nb):
        '''
        Operates on a single batch of data from the validation set. In this step you'd might generate 
        examples or calculate anything of interest like Dice.
        '''
        x, y = batch
        y_hat = self._model.forward(x.to(torch.float))
        loss = self._loss(y_hat,x)
               
        monitor = self._monitor.onValidationStep(self, self._model, x, y, y_hat, loss, {})
        return monitor

    def validation_epoch_end(self, outputs):
        ''' al terminar la validacion se calcula el promedio de la loss de validacion'''

        samples = self._dataModule.val_sample(3)
        monitor = self._monitor.onValidationEpochEnd(self, self._model, samples, outputs, {})
        return monitor

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        '''
        Operates on a single batch of data from the test set. 
        In this step you’d normally generate examples or calculate anything of interest such as accuracy.
        '''
        x, y = batch
        y_hat = self._model.forward(x)
        y_hat,embedding = self._model.forward(x,return_encoding = True)

        #self.tensor_test.append(embedding.cpu().detach().numpy())
        #self.label_test.append(y[0].item())

        loss = self._loss(y_hat,x)

        monitor = self._monitor.onTestStep(self, self._model, x, x, y_hat, loss, {})
        return monitor

    def test_epoch_end(self, outputs):
        monitor = self._monitor.onTestEpochEnd(self, self._model, outputs, {})
        return monitor

    def forward(self, batch):
        '''
        To predict the output of the model
        '''
        x, y = batch
        return self._model.predict(x)
