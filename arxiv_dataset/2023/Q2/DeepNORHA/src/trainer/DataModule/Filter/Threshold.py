
import numpy as np

from .Filter import Filter


class Threshold(Filter):
    '''
    Example:
    - Type: Threshold
      value: .5
    '''
    
    def __init__(self, value=None, **args):
        '''
        Constructor
        '''
        super().__init__(**args)
        self._value = value or .5


    def apply_one(self,  image, i, prepared):
        '''
        Format the input image for the network
        '''

        image = np.where(image > self._value, 1, 0).astype('uint8')

        return image
