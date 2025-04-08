import pytorch_lightning as pl
import torch
import matplotlib.pyplot as plt
import random
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.manifold import TSNE
import seaborn as sns

class ModelTrainerSVDD(pl.LightningModule):
    def __init__(self, model, optimizer, loss, dataModule, monitor, activate_loss=None, **config) -> None:
        super().__init__()
        self._model = model
        self._optimizer = optimizer
        self._loss = loss
        self._dataModule = dataModule
        self._monitor = monitor
        self._config = config
        self._activate_loss = activate_loss


        # assert objective in ('one-class', 'soft-boundary'), "Objective must be either 'one-class' or 'soft-boundary'."
        self.objective = 'one-class'

        # Deep SVDD parameters
        R = 0.0
        c = None
        self.nu = 0.1
        self.R = torch.tensor(R, device=self.device)  # radius R initialized with 0 by default.
        self.c = torch.tensor(c, device=self.device) if c is not None else None
        
        # # Optimization parameters
        # self.warm_up_n_epochs = 10  # number of training epochs for soft-boundary Deep SVDD before radius R gets updated


    def to(self, device):
        super().to(device)

    def configure_optimizers(self):
        '''
        Configuration of the opimizer used in the model
        '''
        cls = getattr(torch.optim, self._optimizer['Type'])
        optimizer = cls(self._model.parameters(), **{k:v for k,v in self._optimizer.items() if k!='Type'})
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
        it is made it for each batch of data
        '''
        x,y = batch
        #x=x.to(torch.float)
        y_hat = self._model.forward(x)  

        if self.c is None:
            print('Initializing center c...')
            self.c = self.init_center_c(batch, self._model)
            print('Center c initialized.')

        scores = torch.sum((y_hat - self.c) ** 2, dim=1)
        loss = torch.mean(scores)

        monitor = self._monitor.onTrainingStep(self, self._model, x, y, y_hat, loss, {})
        return monitor

    def training_epoch_end(self, outputs):
        '''
        Function called at the end of the training loop in one epoch
        outputs: values saved in the return of the training_step
        '''
        self.log('mse_loss',torch.stack([x['loss'] for x in outputs]).mean())
        self._monitor.onTrainingEpochEnd(self, self._model, outputs, {})

    def validation_step(self, batch, batch_nb):
        '''
        Operates on a single batch of data from the validation set. In this step you'd might generate 
        examples or calculate anything of interest like Dice.
        '''
        idx_label_score = []
        x_val,y_val = batch
        y_hat_val = self._model.forward(x_val)
        scores_val = torch.sum((y_hat_val - self.c) ** 2, dim=1)

        idx_label_score += list(zip(y_val[0].cpu().data.numpy().tolist(),
                                    scores_val.cpu().data.numpy().tolist()))
        #x,y = batch
        #y_hat = self._model.forward(x.to(torch.float))
        #scores = torch.sum((y_hat - self.c) ** 2, dim=1)
        loss = torch.mean(scores_val)
               
        #idx_label_score += list(zip(y[0].cpu().data.numpy().tolist(),
        #                            scores.cpu().data.numpy().tolist()))
        labels, scores = zip(*idx_label_score)
        labels = np.array(labels)
        scores = np.array(scores)
        monitor = self._monitor.onValidationStep(self, self._model, y_hat_val, labels, scores, loss, {})
        return monitor

    def validation_epoch_end(self, outputs):
        ''' al terminar la validacion se calcula el promedio de la loss de validacion
        '''
        salidas = torch.cat([x['yhat'] for x in outputs],dim=0)
        etiquetas = torch.cat([x['y'] for x in outputs],dim=0)
        salidas_tse = torch.cat([x['y_hat_val'] for x in outputs],dim=0)
        normales, anormales = [],[]

        for i in range(len(salidas)):
            if etiquetas[i]==0:
                normales.append(salidas[i])
            else:
                anormales.append(salidas[i])
        auc = roc_auc_score(etiquetas, salidas)

        self.log('d_normales',float(sum(normales) / len(normales)))
        self.log('d_anormales',float(sum(anormales) / len(anormales)))
        self.log('d_difference',float(sum(anormales) / len(anormales))-float(sum(normales) / len(normales)))
        self.log('AUC',auc)

        #if self.current_epoch%2==0:

         #   tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
          #  labels = np.array(etiquetas).reshape((139,1))
           # tensors =salidas_tse.cpu().detach().numpy()

            #labels = np.squeeze(labels)
            #tsne_results = tsne.fit_transform(tensors)

            #data = {}
            #data["x"] = tsne_results[:, 0]
            #data["y"] = tsne_results[:, 1]
            #data["z"] = labels

            #sns.scatterplot(x="x", y="y", hue="z", data=data, palette="viridis", legend="full", s=10)
            #plt.savefig('../results/experiment_2b/tsne'+str(self.current_epoch)+'.png')
            #plt.close()

        samples = self._dataModule.val_sample(3)
        monitor = self._monitor.onValidationEpochEnd(self, self._model, samples, outputs, {})
        return monitor

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        '''
        Operates on a single batch of data from the test set. 
        In this step you’d normally generate examples or calculate anything of interest such as accuracy.
        '''
        #x,y = batch
        #y_hat = self._model.forward(x)
        scores = torch.sum((y_hat - self.c) ** 2, dim=1)
        loss = torch.mean(scores)

        monitor = self._monitor.onTrainingStep(self, self._model, x, y, y_hat, loss, {})
        return monitor

    def test_epoch_end(self, outputs):
        monitor = self._monitor.onTestEpochEnd(self, self._model, outputs, {})
        return monitor

    def forward(self, batch):
        '''
        To predict the output of the model
        '''
        x,y = batch
        return self._model.predict(x)


    def init_center_c(self, train_loader, net, eps=0.1):
        """Initialize hypersphere center c as the mean from an initial forward pass on the data.
        """
        n_samples = 0
        c = torch.zeros(net.rep_dim, device=self.device)

        net.eval()
        with torch.no_grad():
            inputs, _= train_loader
            #inputs = inputs.to(self.device)
            inputs = [inputs[0].to(self.device),inputs[1].to(self.device),inputs[2].to(self.device)]  #$$$#
            outputs = net(inputs)
            n_samples += outputs.shape[0]
            c += torch.sum(outputs, dim=0)
        net.train()
        c /= n_samples
        # If c_i is too close to 0, set to +-eps. Reason: a zero unit can be trivially matched with zero weights.
        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c > 0)] = eps
        return c


