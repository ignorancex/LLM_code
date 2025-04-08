from .Monitor import Monitor

class MetricLogger(Monitor):
    """
    Example:

    - Type: MetricLogger
      metric: dice
    """
    def __init__(self, metric = None, metrics=None, **args) -> None:
        super().__init__(**args)
        self._metrics = (metrics or []) + [metric] or []

    def onTrainingEpochEnd(self, trainer, model, outputs, results):
        for metric in self._metrics:
            if metric in results:
                trainer.log(metric,results[metric])
        return super().onTrainingEpochEnd(trainer, model, outputs, results)
        
    def onValidationEpochEnd(self, trainer, model, samples, outputs, results):
        for metric in self._metrics:
            if metric in results:
                trainer.log(metric,results[metric])
        return super().onValidationEpochEnd(trainer, model, samples, outputs, results)

    def onTestEpochEnd(self, trainer, model, outputs, results):
        for metric in self._metrics:
            if metric in results:
                trainer.log(metric,results[metric])
        return super().onTestEpochEnd(trainer, model, outputs, results)