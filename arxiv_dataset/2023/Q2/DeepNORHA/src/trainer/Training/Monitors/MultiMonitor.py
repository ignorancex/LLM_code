from .Monitor import Monitor

class MultiDict(dict):
    def update(self, other):
        for k, v in other.items():
            if k in self and isinstance(self[k], dict) and isinstance(v, dict):
                self[k].update(v)
            else:
                self[k] = v


class MultiMonitor(Monitor):
    def __init__(self, monitors, **args) -> None:
        super().__init__(**args)
        self._monitors = [
            self.trainFactory.getMonitor(monitor) for monitor in monitors
        ]

    def onTrainingStep(self, trainer, model, x, y, y_hat, loss, results):
        r = MultiDict(results)
        for monitor in self._monitors:
            r = monitor.onTrainingStep(trainer, model, x, y, y_hat, loss, r)
        return r

    def onTrainingEpochEnd(self, trainer, model, outputs, results):
        r = results or MultiDict(results)
        for monitor in self._monitors:
            r = monitor.onTrainingEpochEnd(trainer, model, outputs, r)
        return r

    def onValidationStep(self, trainer, model, x, y, y_hat, loss, results):
        r = MultiDict(results)
        for monitor in self._monitors:
            r = monitor.onValidationStep(trainer, model, x, y, y_hat, loss, r)
        return r

    def onValidationEpochEnd(self, trainer, model, samples, outputs, results):
        r = MultiDict(results)
        for monitor in self._monitors:
            r = monitor.onValidationEpochEnd(trainer, model, samples, outputs, r)
        return r

    def onTestStep(self, trainer, model, x, y, y_hat, loss, results):
        r = MultiDict(results)
        for monitor in self._monitors:
            r = monitor.onTestStep(trainer, model, x, y, y_hat, loss, r)
        return r

    def onTestEpochEnd(self, trainer, model, outputs, results):
        r = MultiDict(results)
        for monitor in self._monitors:
            r = monitor.onTestEpochEnd(trainer, model, outputs, r)
        return r