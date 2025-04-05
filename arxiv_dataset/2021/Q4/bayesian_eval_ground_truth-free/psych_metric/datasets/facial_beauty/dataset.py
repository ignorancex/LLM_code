"""Dataset class handler for facial beauty 2018 data."""
import os

import cv2
import h5py
import numpy as np
import pandas as pd

from psych_metric.datasets.base_dataset import BaseDataset

try:
    ROOT = os.environ['ROOT']
    HERE = os.path.join(ROOT, 'psych_metric/datasets/facial_beauty/facial_beauty_data/')
except:
    HERE = None


class FacialBeauty(BaseDataset):
    """class that loads and serves data from facial beauty 2018

    Attributes
    ----------
    dataset : str
        Name of specific dataset
    task_type : str
        The type of learning task the dataset is intended for. This can be one
        of the following: 'regression', 'binary_classification', 'classification'
    df : pandas.DataFrame
        Data Frame containing annotations
    label_set : set
        Set containing the complete original labels
    label_encoder : {str: sklearn.preprocessing.LabelEncoder}
        Dict of column name to Label Encoder for the labels. None if no
        encodings used.
    sparse_matrix : bool
        Dataframe uses datafile structure if True, uses sparse matrix format if
        False. Default value is False
    """

    datasets = frozenset([
        'FacialBeauty',
        'All_Ratings',
        'Asian_Female',
        'Asian_Male',
        'Caucasian_Female',
        'Caucasian_Male',
    ])

    def __init__(self, dataset='All_Ratings', dataset_filepath=None, encode_columns=None, sparse_matrix=False):
        """initialize class by loading the data. No ground truth available.

        Parameters
        ----------
        dataset : str
            the name of one of the subdatasets corresponding to file name
        encode_columns : list, optional
            Encodes columns provided as list of str; dataframe uses raw values
            by default.
        sparse_matrix : bool, optional
            Convert the data into a dataframe with the sparse matrix structure
        """
        self._check_dataset(dataset, FacialBeauty.datasets)
        self.dataset = 'All_Ratings' if dataset == 'FacialBeauty' else dataset
        # NOTE this could be percieved as a regression task, or at least an ordinal task
        self.task_type = 'regression'  # regression or ordinal due to ints.

        if dataset_filepath is None:
            if HERE is None or 'facial_beauty_data' not in HERE:
                raise ValueError('A path to the dataset file was not provided either by the `dataset_filepath` parameter or by the ROOT environment variable. Global variable HERE is `%s`. It is recommended to use the `dataset_filepath` parameter to provide the filepath.' % HERE)
            dataset_filepath = HERE
        self.data_dir = dataset_filepath

        if not isinstance(sparse_matrix, bool):
            raise TypeError('sparse_matrix parameter must be a boolean.')
        self.sparse_matrix = sparse_matrix

        # Read in and save data
        annotation_file = os.path.join(dataset_filepath, self.dataset + '.csv')
        self.df = pd.read_csv(annotation_file)
        self.df.columns = ['worker_id', 'sample_id', 'worker_label', 'original_rating']

        # Save labels set
        # self.label_set = frozenset((1,2,3,4,5)) # NOTE treating this as regression task
        self.label_set = None

        if encode_columns is True:
            encode_columns = {'sample_id'}

        # Encode the labels and data if desired
        self.label_encoder = None if encode_columns is None else self.encode_labels(encode_columns)

        # Restructure dataframe into a sparse matrix
        if sparse_matrix:
            self.df = self.convert_to_sparse_matrix(self.df)

    # TODO get_image
    def get_image(self):
        raise NotImplementedError

    def load_images(self, image_dir=None, train_filenames=None, annotations=None, ground_truth=None, majority_vote=False, img_shape=None, color=cv2.IMREAD_COLOR, output=None, num_tfrecords=1, normalize=True):
        """Load the images and optionally crop  by the bounding box files.

        Parameters
        ----------
        image_dir : str
            filepath to directory containing images
        train_filenames : str
            filepath to file containing filenames of images
        ground_truth : str
            filepath to file containing ground truth label
        majority_vote : str
            filepath to file containing majority votes of annotations
        img_shape : tuple of ints
            The shape of the images to be reshaped to if provided, otherwise no
            resizing is done.
        color : int
            The imread method to use. Defaults to color reading.

        Returns
        -------
            images and samples if output is not provided, otherwise saves the
            tfrecords to file and returns None.
        """
        if image_dir is None:
            image_dir = os.path.join(
                self.data_dir,
                'Images',
            )
        elif not isinstance(image_dir, str) or not os.path.exists(image_dir):
            raise IOError(f'The path `{image_dir}` does not exist.')

        # TODO need to put annotations in "sparse" matrix format.
        if self.is_in_annotation_list_format():
            self.annotation_list_to_sparse_matrix(inplace=True)

        # load train_filenames
        if majority_vote:
            # NOTE this is basically argmax, lowest mode is selected as vote.
            # TODO weighted mean of class values then round to closest?
            samples = self.df.mode(axis=1)[0]
            samples.name = 'majority_vote'
        else:
            samples = None

        filename_idx = {idx: i for i, idx in enumerate(self.df.index)}

        images = np.empty(len(self.df)).tolist()
        if output and isinstance(output, str):
            shape = np.empty(len(samples)).tolist()

        if os.path.isdir(image_dir):
            for filename in os.listdir(image_dir):
                if filename in filename_idx:
                    img = cv2.imread(
                        os.path.join(image_dir, filename),
                        color,
                    ).astype(np.uint8)

                    if img_shape and img.shape != img_shape:
                        img = cv2.resize(img, img_shape, interpolation=cv2.INTER_CUBIC)

                    if normalize:
                        img = img / 255.0

                    images[filename_idx[filename]] = img
                    if output and isinstance(output, str):
                        shape[filename_idx[filename]] = img.shape
                else:
                    print(f'{filename} not in filename_idx')
            images = np.stack(images)
        elif os.path.isfile(image_dir):
            with h5py.File(image_dir, 'r') as h5f:
                images = h5f['images'][:]
        else:
            raise IOError(f'The path `{image_dir}` does not exist...')

        # samples.drop(columns=['index', 'filename'], inplace=True)

        if samples is not None:
            return images, samples
        return images

    def __len__(self):
        """ get size of dataset

        Returns
        -------
        int
            number of annotations(rows) in dataset. If sparse matrix, number of
            samples
        """
        return len(self.df)

    def __getitem__(self, i):
        """ get specific row from dataset

        Returns
        -------
        dict:
            {header: value, header: value, ...}
        """
        row = self.df.iloc[i]
        return dict(row)
