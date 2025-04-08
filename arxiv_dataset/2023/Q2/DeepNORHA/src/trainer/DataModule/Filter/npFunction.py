
import numpy as np

from .Filter import Filter


class npFunction(Filter):
    '''
    Example:
        - Type: npFunction
          fun: expand_dims
          axis: 2
    '''
    
    def __init__(self, fun, fun_args=None, **args):
        '''
        Constructor
        '''
        super().__init__(**args)
        self._fun = getattr(np, fun)
        self._fun_args = fun_args or {}


    def apply_one(self, image, i, prepared):
        '''
        Apply function to each image 
        '''
        return self._fun(image, **self._fun_args)
