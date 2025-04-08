from email.policy import strict
import os
from typing import Iterable, Tuple
import numpy as np

import nibabel as ni

from torch.utils.data import Dataset, DataLoader
import random
import torch

from configparser import ConfigParser

class Dataset(torch.utils.data.Dataset):
    '''
    Dataset object
    '''
    
    def __init__(self, dataFactory, split, input_path, filenames, pipe=None, len=None, **args):
        '''
        Constructor
        '''
        self.dataFactory = dataFactory

        # assign the split (training, validation or test)        
        assert split in ['training', 'validation', 'test'], "Unknown split {}. It should be training, validation or test.".format(split)
        self.split = split
        # assign the names of the files that will be loaded
        self.filenames = filenames.split(",")
        # print('DATASET', filenames)
        # assign the input folder
        self.input_path = input_path
        
        self._pipe = dataFactory.getPipe(pipe)

        self._len = len if split != 'test' else None
        super().__init__()

    def __len__(self):
        '''
        Return the number of elements
        '''
        return self._len or len(self.filenames)

    def __getitem__(self, element):
        '''
        Get an item for the batch
        '''
        if self._len:
            element = random.choice(self.filenames)

        else:
            element = self.filenames[element]
        
        return self.get(element)

    def get(self, element, test=None):
        
        test = test or self.split == 'test'
        for filter in self._pipe:
            element = filter(element, test)
        return element

    def sample(self, size=None) -> Tuple[Iterable, Iterable[str], Iterable]:
        '''
        Sample elements
        Returns a list of elements without deformations for testing
        '''
        # All elements as default
        size = size or len(self.filenames)
        
        idx = list(range(len(self.filenames)))
        if size < len(idx):
            # If not all elements are sampled, sample randomly
            idx = random.sample(idx, size)
            
        filenames = [self.filenames[i] for i in idx]

        return [
            (filename, self.get(filename, test=True))
            for filename in filenames
        ]
