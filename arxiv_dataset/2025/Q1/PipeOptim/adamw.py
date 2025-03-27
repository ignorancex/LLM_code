# Copyright (c) Microsoft Corporation.
#           (c) I-Ching Tseng
# Licensed under the MIT license.

from torch.optim.optimizer import required
from optimizer import OptimizerWithWeightPrediction


class ADAMWWithPipeOptim(OptimizerWithWeightPrediction):
    """
    Adam optimizer with spectrain.
    """

    def __init__(self,
                 modules,
                 master_parameters,
                 model_parameters,
                 loss_scale,
                 num_versions,
                 lr=required,
                 betas=(0.9, 0.999),
                 weight_decay=0,
                 verbose_freq=0,
                 partitions=1):
        super(ADAMWWithPipeOptim, self).__init__(
            optim_name='AdamW',
            modules=modules,
            master_parameters=master_parameters,
            model_parameters=model_parameters,
            loss_scale=loss_scale,
            num_versions=num_versions,
            lr=lr,
            betas=betas,
            weight_decay=weight_decay,
            verbose_freq=verbose_freq,
            partitions=partitions,
        )