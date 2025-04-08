

class Monitor(object):
    def __init__(self, trainFactory, **args) -> None:
        super().__init__()
        self.trainFactory = trainFactory

    def onTrainingStep(self, trainer, model, x, y, y_hat, loss, results):
        return results

    def onTrainingEpochEnd(self, trainer, model, outputs, results):
        return results

    def onValidationStep(self, trainer, model, x, y, y_hat, loss, results):
        return results

    def onValidationEpochEnd(self, trainer, model, samples, outputs, results):
        return results

    def onTestStep(self, trainer, model, x, y, y_hat, loss, results):
        return results

    def onTestEpochEnd(self, trainer, model, outputs, results):
        return results