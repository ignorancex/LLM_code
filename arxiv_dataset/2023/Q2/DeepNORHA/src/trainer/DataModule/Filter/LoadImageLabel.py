import os

import numpy as np
from PIL import Image

from .Filter import Filter

import pickle

class LoadImageLabel(Filter):
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
        
        with open('/home/ddeangeli/deepsvdd/data/labels/all_labels_anomaly_detection.pkl', 'rb') as fp:
            dictLabels = pickle.load(fp)  

        label = dictLabels.get(filename)
        return label

