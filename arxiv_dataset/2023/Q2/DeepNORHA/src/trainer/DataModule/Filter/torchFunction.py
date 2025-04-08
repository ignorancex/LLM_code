
import torch

from .Filter import Filter


class torchFunction(Filter):
    '''
    Example:
        - Type: torchFunction
          fun: sigmoid
    '''
    
    def __init__(self, fun, fun_args=None, **args):
        '''
        Constructor
        '''
        super().__init__(**args)
        self._fun = getattr(torch, fun)
        self._fun_args = fun_args or {}


    def apply(self, images):
        '''
        Reduce images 
        '''
        return self._fun(images, **self._fun_args)
