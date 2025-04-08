
from .MultiMonitor import MultiMonitor, MultiDict

class MultiTaskMonitor(MultiMonitor):
    def onTrainingStep(self, trainer, model, x, ys, y_hats, losses, results):
        r = MultiDict(results)
        r = self._monitors[0].onTrainingStep(trainer, model, x, ys[-1], y_hats[-1], losses[0], r)
        for y, y_hat, loss, monitor in zip(ys, y_hats, losses[1:], self._monitors[1:]):
            r = monitor.onTrainingStep(trainer, model, x, y, y_hat, loss, r)
        return r

    def onValidationStep(self, trainer, model, x, ys, y_hats, losses, results):
        r = MultiDict(results)
        r = self._monitors[0].onValidationStep(trainer, model, x, ys[-1], y_hats[-1], losses[0], r)
        for y, y_hat, loss, monitor in zip(ys, y_hats, losses[1:], self._monitors[1:]):
            r = monitor.onValidationStep(trainer, model, x, y, y_hat, loss, r)
        return r

    def onTestStep(self, trainer, model, x, ys, y_hats, losses, results):
        r = MultiDict(results)
        r = self._monitors[0].onTestStep(trainer, model, x, ys[-1], y_hats[-1], losses[0], r)
        for y, y_hat, loss, monitor in zip(ys, y_hats, losses[1:], self._monitors[1:]):
            r = monitor.onTestStep(trainer, model, x, y, y_hat, loss, r)
        return r