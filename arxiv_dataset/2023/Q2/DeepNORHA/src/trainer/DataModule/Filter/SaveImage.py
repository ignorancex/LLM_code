import os

import numpy as np
from PIL import Image

from .Filter import Filter


class SaveImage(Filter):
    '''
    Stores an image
    Example:
    - Type: SaveImage
      subfolder: myresult
    '''
    
    def __init__(self, output_path, subfolder='', **args):
        '''
        Constructor
        '''
        super().__init__(**args)
        self._output_path = output_path
        self._subfolder = subfolder

    def apply(self, element):
        '''
        Store image
        '''
        name, image = element
        image = (image*255).astype(np.uint8)
        image = Image.fromarray(image)
        path = os.path.join(self._output_path, self._subfolder, f"{name}.png")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        image.save(path)

        return element

