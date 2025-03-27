# Copyright (c) Microsoft Corporation.
#           (c) I-Ching Tseng
# Licensed under the MIT license.

from torch.optim.optimizer import required

from optimizer import OptimizerWithWeightPrediction

class RMSpropWithWeightPrediction(OptimizerWithWeightPrediction):
    """
    SGD optimizer with weight stashing.
    """
    def __init__(self, modules, master_parameters, model_parameters,
                 loss_scale, num_versions, lr=required, momentum=0,
                 alpha=0.99, weight_decay=0, verbose_freq=0,
                 partitions=1):
        super(RMSpropWithWeightPrediction, self).__init__(
            optim_name='RMSprop',
            modules=modules, master_parameters=master_parameters,
            model_parameters=model_parameters, loss_scale=loss_scale,
            num_versions=num_versions, lr=lr, momentum=momentum,
            alpha=alpha, weight_decay=weight_decay,
            verbose_freq=verbose_freq,
            partitions=partitions,
        )