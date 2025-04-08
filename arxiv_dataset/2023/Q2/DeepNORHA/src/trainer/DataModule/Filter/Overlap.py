import cv2

from .Filter import Filter


class Overlap(Filter):
    '''
    Example:
        - Type: Overlap
    '''
    
    def __init__(self, alpha=None, **args):
        '''
        Constructor
        '''
        super().__init__(**args)
        self._alpha = alpha or .5

    def apply(self, images):
        '''
        Overlap images 
        '''
        return cv2.addWeighted(images[0], self._alpha, images[1], 1-self._alpha, 0)
