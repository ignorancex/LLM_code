import numpy as np

from .Filter import Filter


class ToCategorical(Filter):
    '''
    Adds a dimension to a np.array with a one hot encoding

    Example:
    - Type: ToCategorical
      num_classes: 2
    '''
    
    def __init__(self, num_classes, **args):
        '''
        Constructor
        '''
        super().__init__(**args)
        self._num_classes = num_classes
        

    def apply_one(self,  image, i, prepared):
        '''
        '''
        return np.eye(self._num_classes)[image]