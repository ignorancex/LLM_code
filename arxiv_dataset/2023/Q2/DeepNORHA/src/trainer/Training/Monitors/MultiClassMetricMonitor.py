import numpy as np
import torch
from DataModule import DataFactory

from . import metrics
from .MetricMonitor import MetricMonitor


class MultiClassMetricMonitor(MetricMonitor):
    """
    Apply metric to each class.

    Examples:
    - Type: MultiClassMetricMonitor
      name: Classes
      metric: f1_score_for_bifurcations
      pipe:
      - ...
      - ...
      classes:
      - First
      - Second
      
    - Type: MultiClassMetricMonitor
      name: Classes
      metric: f1_score_for_bifurcations
      pipe:
      - ...
      - ...
      classes:
      - Background
      - One
      - Two
      ignore: Background
      
    - Type: MultiClassMetricMonitor
      name: Classes
      metric: f1_score_for_bifurcations
      pipe:
      - ...
      - ...
      classes:
      - First
      - One
      - Two
      ignore:
      - One
      - Two
    """
    def __init__(self, classes, ignore=None, **args):
        super().__init__(**args)
        self._classes = classes
        self._ignore = ignore or []
        
    def onValidationStep(self, trainer, model, x, y, y_hat, loss, results):
        elements = (y.detach().cpu(), y_hat.detach().cpu())
        for filter in self._pipe:
            elements = filter(elements)
        for class_, *element in zip(self._classes, *elements):
            if class_ in self._ignore:
                continue
            metric = self._fun(*element, **self._fun_args)
            results.update({f'{self._metric}{self._name}{class_}': metric})
        return results

    def onValidationEpochEnd(self, trainer, model, samples, outputs, results):
        avgs = []
        for class_ in self._classes:
            if class_ == self._ignore:
                continue
            avg = np.mean([x[f'{self._metric}{self._name}{class_}']
                                for x in outputs])
            results.update({f'{self._metric}{self._name}{class_}': avg})
            results.update({f'{self._metric}': {f'{self._name}{class_}': avg}})
            avgs.append(avg)
        avgs = np.mean(avgs)
        results.update({f'{self._metric}{self._name}': avgs})
        return results

    def onTestStep(self, trainer, model, x, y, y_hat, loss, results):
        elements = (y.detach().cpu(), y_hat.detach().cpu())
        for filter in self._pipe:
            elements = filter(elements)
        for class_, *element in zip(self._classes, *elements):
            if class_ in self._ignore:
                continue
            metric = self._fun(*element, **self._fun_args)
            results.update({f'{self._metric}{self._name}{class_}': metric})
        return results

    def onTestEpochEnd(self, trainer, model, outputs, results):
        avgs = []
        for class_ in self._classes:
            if class_ == self._ignore:
                continue
            avg = np.mean([x[f'{self._metric}{self._name}{class_}']
                                for x in outputs])
            results.update({f'{self._metric}{self._name}{class_}': avg})
            results.update({f'{self._metric}': {f'{self._name}{class_}': avg}})
            avgs.append(avg)
        avgs = np.mean(avgs)
        results.update({f'{self._metric}{self._name}': avgs})
        return results