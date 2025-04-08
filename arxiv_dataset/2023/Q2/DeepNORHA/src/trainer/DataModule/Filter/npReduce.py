
import numpy as np

from .Filter import Filter


class npReduce(Filter):
    '''
    Example:
        - Type: npReduce
          fun: logical_or
    '''
    
    def __init__(self, fun, fun_args = None, type = None, **args):
        '''
        Constructor
        '''
        super().__init__(**args)
        self._fun = getattr(np, fun)
        self._type = getattr(np, type or 'uint8')
        self._fun_args = fun_args or {}

    def apply(self, images):
        '''
        Reduce images 
        '''
        return self._fun.reduce(images, **self._fun_args).astype(self._type)
