"""
data_generator.py

This module handles the creation of data generators for efficient data loading and preprocessing during training.

Class:
    DataGenerator: A data generator for efficient data loading and preprocessing during training.

Methods:
    __init__(self, params, mode='dev_train'): Initializes the DataGenerator instance.
    __getitem__(self, item): Returns the data for a given index.
    __len__(self): Returns the number of data points.
    get_feature_files(self): Collects the paths to the feature files based on the selected folds
    get_folds(self): Returns the folds for the given data split.

Author: Parthasaarathy Sudarsanam, Audio Research Group, Tampere University
Date: February 2025
"""

import os
import torch
import glob
from torch.utils.data.dataset import Dataset


class DataGenerator(Dataset):
    def __init__(self, params, mode='dev_train', transform=None):
        """
        Initializes the DataGenerator instance.
        Args:
            params (dict): Parameters for data generation.
            mode (str): data split ('dev_train', 'dev_test').
        """

        super().__init__()
        self.params = params
        self.mode = mode
        self.root_dir = params['root_dir']
        self.feat_dir = params['feat_dir']
        self.transform = transform

        self.folds = self.get_folds()

        self.audio_files, self.label_files = self.get_feature_files()

    def __getitem__(self, item):
        """
        Returns the data for a given index.
        Args:
            item (int): Index of the data.
        Returns:
            tuple: A tuple containing audio features and labels.
        """
        audio_file = self.audio_files[item]
        label_file = self.label_files[item]
        audio_features = torch.load(audio_file)
        labels = torch.load(label_file)
        if not self.params['multiACCDOA']:
            mask = labels[:, :self.params['nb_classes']]
            mask = mask.repeat(1, 4)
            labels = mask * labels[:, self.params['nb_classes']:]

        # no need for on/off labels for audio only task
        if self.params['multiACCDOA']:
            labels = labels[:, :, :-1, :]
        else:
            labels = labels[:, :-self.params['nb_classes']]

        # Apply augmentations if any
        if self.transform is not None:
            audio_features = self.transform(audio_features)

        return audio_features, labels

    def __len__(self):
        """
        Returns the number of data points.
        Returns:
            int: Number of data points.
        """
        return len(self.audio_files)

    def get_feature_files(self):
        """
        Collects the paths to the feature and label files based on the selected folds
        Returns:
            tuple: A tuple containing lists of paths to audio feature files, video feature files, and processed label files.
        """
        audio_files, label_files = [], []

        # Loop through each fold and collect files
        for fold in self.folds:
            audio_files += glob.glob(os.path.join(self.feat_dir, f'stereo_dev_normalized/{fold}*.pt'))
            label_files += glob.glob(os.path.join(self.feat_dir, 'metadata_dev{}/{}*.pt'.format('_adpit' if self.params['multiACCDOA'] else '', fold)))

        # Sort files to ensure corresponding audio, video, and label files are in the same order
        audio_files = sorted(audio_files, key=lambda x: x.split('/')[-1])
        label_files = sorted(label_files, key=lambda x: x.split('/')[-1])

        # Return the appropriate files
        return audio_files, label_files


    def get_folds(self):
        """
        Returns the folds for the given data split
        Returns:
            list: List of folds.
        """
        if self.mode == 'dev_train':
            if self.params['finetune']:
                return ['fold3']
            else:
                return self.params['dev_train_folds']  # fold 1, fold 3
        elif self.mode == 'dev_test':
            return self.params['dev_test_folds']  # fold 4
        else:
            raise ValueError(f"Invalid mode: {self.mode}. Choose from ['dev_train', 'dev_test'].")