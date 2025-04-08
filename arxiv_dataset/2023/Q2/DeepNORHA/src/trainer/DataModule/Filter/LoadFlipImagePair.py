import os

import numpy as np
from PIL import Image
import torch

from .Filter import Filter

import nibabel as nb

class LoadFlipImagePair(Filter):
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

        if 'left' in filename:
            image = np.flip(image,axis=0).copy()

            new_filename = filename.replace("left", "right" )
            imagepair = nb.load(os.path.join(self._input_path, self._subfolder, new_filename + '.nii'))
            imagepair = np.array(imagepair.get_fdata())


        else:
            new_filename = filename.replace("right","left" )
            imagepair = nb.load(os.path.join(self._input_path, self._subfolder, new_filename + '.nii'))
            imagepair = np.array(imagepair.get_fdata())
            imagepair = np.flip(imagepair,axis=0).copy()
        # print(filename,new_filename)

        # print('loadflipimagepair')
        image = torch.tensor(image)
        imagepair = torch.tensor(imagepair)

        # print('tensorflipimagepair')
        image = torch.cat((torch.unsqueeze(image, 0),torch.unsqueeze(imagepair, 0)), 0)
        # print(image.shape)
        return image

