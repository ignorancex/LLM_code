import logging
import torch.nn as nn
import numpy as np
from .EncoderDecoderNetwork import EncoderDecoderNetwork

class BaseNet(EncoderDecoderNetwork):
    """Base class for all neural networks."""

    def __init__(self, **config):

        ''' 
        Constructor.
        -------
        '''
        super().__init__(**config)

        self.name = 'dulito'

        self.rep_dim = self.rep_dim  # representation dimensionality, i.e. dim of the last layer
        self.hipo_diff = self.hipo_diff
        self.fc_out = self.fc_out
        self.VolumeDifference = self.VolumeDifference

    def forward(self, *input):
        """
        Forward pass logic
        :return: Network output
        """
        raise NotImplementedError

    def summary(self):
        """Network summary."""
        net_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in net_parameters])
        self.logger.info('Trainable parameters: {}'.format(params))
        self.logger.info(self)
