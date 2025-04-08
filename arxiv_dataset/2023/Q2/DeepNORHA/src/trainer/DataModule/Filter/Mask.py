
import torch

from .Filter import Filter


class Mask(Filter):
    '''
    Example:
    - Type: Mask
    '''

    def apply_one(self,  image, i, prepared):
        image[image>0] = 1

        return image