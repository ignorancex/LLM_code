
import numpy as np

from .Filter import Filter


class Stack(Filter):
    '''
    Dataset object for image datasets
    '''
    
    def __init__(self, axis=-1, **args):
        '''
        Constructor
        '''
        super().__init__(**args)
        self._axis = axis


    def apply(self, images):
        '''
        Stack images
        '''
        image = np.concatenate(images,axis=self._axis)

        return image
