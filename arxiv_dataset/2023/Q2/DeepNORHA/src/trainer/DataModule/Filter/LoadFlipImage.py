import os

import numpy as np
from PIL import Image

from .Filter import Filter

import nibabel as nb

class LoadFlipImage(Filter):
    '''
    Dataset object for image datasets
    '''
    
    def __init__(self, input_path, subfolder='', **args):
        '''
        Constructor
        '''
        super().__init__(**args)
        self._input_path = input_path
        self._subfolder = subfolder

    def apply_one(self,  filename, i, prepared):
        '''
        Read an image
        '''
        # load the image using nibabel
        # print(self._input_path)
        # print('filename',filename)
        image = nb.load(os.path.join(self._input_path, self._subfolder, filename + '.nii'))
        image = np.array(image.get_fdata())
        # correct dimensions if necessary by adding the color band
        # if len(image.shape)<5:
        #   image = np.expand_dims(image, axis=2)
        if 'left' in filename:
            image = np.flip(image,axis=0).copy()
            # print("GOTCHA")
        
        return image

