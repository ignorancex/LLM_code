from .Monitor import Monitor
import torch
import numpy as np

from DataModule import DataFactory

class SamplePlot(Monitor):
    def __init__(self, pipe=None, **args) -> None:
        super().__init__(**args)
        dataFactory = DataFactory()

        self._pipe = dataFactory.getPipe(pipe)
        self.VolumeDifference = args.get('VolumeDifference', True)

    def onValidationEpochEnd(self, trainer, model, samples, outputs, results):

        for name, (x,y) in samples:
            # x = torch.unsqueeze(x, dim=0)
            try:
                pred = model.test(x)

                element = (x,pred[0])
                for filter in self._pipe:
                    element = filter(element)

            except:
                element = x

            if self.VolumeDifference:
                element = element[0] #$$$#

            sagital = np.concatenate((element[0][32,:,:],element[1][32,:,:]),axis=1)
            axial = np.concatenate((element[0][:,32,:],element[1][:,32,:]),axis=1)
            coronal = np.concatenate((element[0][:,:,32],element[1][:,:,32]),axis=1)
            element = np.concatenate((sagital,axial,coronal),axis=0)
            # print(element.shape)
            for logger in trainer.loggers:
                if hasattr(logger.experiment, 'image'):
                    logger.experiment.image(element, f"{name}", opts={'title': name})

        return super().onValidationEpochEnd(trainer, model, samples, outputs, results)
