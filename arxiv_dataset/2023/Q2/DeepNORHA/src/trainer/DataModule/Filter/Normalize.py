
import torch

from .Filter import Filter


class Normalize(Filter):
    '''
    Example:
    - Type: Normalize
      max: 1
      min: 0
    '''
    
    def __init__(self, max, min=None, **args):
        '''
        Constructor
        '''
        super().__init__(**args)
        self._max = max
        self._min = min or 0


    def apply_one(self,  image, i, prepared):
        '''
        Format the input image for the network
        '''

        image = (image - self._min) / (self._max - self._min)

        return image