from .Monitor import Monitor
import torch

class LossMonitor(Monitor):
    def __init__(self, name=None, **args):
        super().__init__(**args)
        self._name = name or ""

    def onTrainingStep(self, trainer, model, x, y, y_hat, loss, results):
        results = super().onTrainingStep(trainer, model, x, y, y_hat, loss, results)
        results.update({f'loss{self._name}':loss})
        return results
        
    def onTrainingEpochEnd(self, trainer, model, outputs, results):
        results = super().onTrainingEpochEnd(trainer, model, outputs, results)
        avg_loss = torch.stack([x[f'loss{self._name}'] for x in outputs]).mean()
        results.update({'Loss':{f'{self._name}_train':avg_loss}})
        return results
        
    def onValidationStep(self, trainer, model, x, y, y_hat, loss, results):
        results = super().onValidationStep(trainer, model, x, y, y_hat, loss, results)
        results.update({f'loss{self._name}':loss})
        if y_hat[1].shape == ():
           results.update({f'yhat{self._name}':torch.from_numpy(y_hat)})
           results.update({f'y{self._name}':torch.from_numpy(y)})
           results.update({f'y_hat_val{self._name}':x})
        return results

    def onValidationEpochEnd(self, trainer, model, samples, outputs, results):
        results = super().onValidationEpochEnd(trainer, model, samples, outputs, results)
        avg_loss = torch.stack([x[f'loss{self._name}'] for x in outputs]).mean()
        results.update({'Loss':{f'{self._name}_val':avg_loss}})
        return results
