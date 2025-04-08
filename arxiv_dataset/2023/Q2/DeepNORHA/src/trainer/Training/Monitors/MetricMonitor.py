import numpy as np
from DataModule import DataFactory

from . import metrics
from .Monitor import Monitor


class MetricMonitor(Monitor):
    def __init__(self, metric, name = None, fun_args = None, pipe = None, **args):
        super().__init__(**args)
        self._metric = metric
        self._fun = getattr(metrics, metric)
        self._fun_args = fun_args or {}
        self._name = name or ""
        dataFactory = DataFactory()
        self._pipe = dataFactory.getPipe(pipe)
        print('MetricMonitor.py',self._metric)

    def onValidationStep(self, trainer, model, x, y, y_hat, loss, results):
        results = super().onValidationStep(trainer, model, x, y, y_hat, loss, results)
        element = (y.detach().cpu(), y_hat.detach().cpu())
        for filter in self._pipe:
            element = filter(element)
        metric = self._fun(*element, **self._fun_args)
        results.update({f'{self._metric}{self._name}': metric})
        return results

    def onValidationEpochEnd(self, trainer, model, samples, outputs, results):
        results = super().onValidationEpochEnd(
            trainer, model, samples, outputs, results)
        avg = np.array([x[f'{self._metric}{self._name}']
                           for x in outputs]).mean()
        results.update({f'{self._metric}': {f'{self._name}': avg}})
        results.update({f'{self._metric}{self._name}': avg})
        return results

    def onTestStep(self, trainer, model, x, y, y_hat, loss, results):
        results = super().onTestStep(trainer, model, x, y, y_hat, loss, results)
        element = (y.detach().cpu(), y_hat.detach().cpu())
        for filter in self._pipe:
            element = filter(element)
        metric = self._fun(*element, **self._fun_args)
        results.update({f'{self._metric}{self._name}': metric})
        return results

    def onTestEpochEnd(self, trainer, model, outputs, results):
        results = super().onTestEpochEnd(trainer, model, outputs, results)
        avg = np.array([x[f'{self._metric}{self._name}']
                           for x in outputs]).mean()
        results.update({f'{self._metric}': {f'{self._name}': avg}})
        results.update({f'{self._metric}{self._name}': avg})
        return results
