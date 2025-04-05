"""Dataset class handler for Dog classification."""
from datetime import datetime
import logging
import os
import xml.etree.ElementTree as ET

import cv2
#import numpy as np
import pandas as pd
import tensorflow as tf

from psych_metric.datasets.base_dataset import BaseDataset

try:
    ROOT = os.environ['ROOT']
    HERE = os.path.join(ROOT, 'psych_metric/datasets/dogs/')
except KeyError:
    HERE = None

class Dogs(BaseDataset):
    """class that loads and serves data from truth inference survey 2017

    Attributes
    ----------
    task_type : str
        The type of learning task the dataset is intended for. This can be one
        of the following: 'regression', 'binary_classification', 'classification'
    images:
    labels:
    label_set : set
        Set containing the complete original labels.
    label_encoder : {str: sklearn.preprocessing.LabelEncoder}
        Dict of column name to Label Encoder for the labels. None if no
        encodings used.
    """
    def __init__(self, dataset=None, dataset_filepath=None, encode_columns=None, ground_truth=False, tf_load=False, preprocess=False):
        """initialize class by loading the data

        Parameters
        ----------
        dataset : str
            the name of one of the subdatasets corresponding to file name
        encode_columns : list, optional
            Encodes columns provided as list of str; dataframe uses raw values
            by default.
        """
        self._check_dataset(dataset, TruthSurvey2017.datasets)
        self.dataset = dataset

        if dataset_filepath is None:
            if HERE is None or 'images' not in HERE:
                raise ValueError(f"""A path to the dataset file was not provided
                either by the `dataset_filepath` parameter or by the ROOT
                environment variable. Global variable HERE is `{HERE}`. It is
                recommended to use the `dataset_filepath` parameter to provide
                the filepath.""")
            dataset_filepath = HERE
        self.data_dir = dataset_filepath

        self.task_type = 'classification'

        if tf_load:

        else:

    def load_file(dataset_filepath, ground_truth=False):
        # Read in and save data
        annotation_file = os.path.join(dataset_filepath,'answer.csv')
        self.df = pd.read_csv(annotation_file)
        self.df.columns = ['sample_id', 'worker_id', 'worker_label']

        labels_file = os.path.join(dataset_filepath, 'truth.csv')
        ground_truth_df = pd.read_csv(labels_file)
        ground_truth_df.columns = ['sample_id', 'worker_label']

        self.label_set = set(ground_truth_df['worker_label'].unique())

        # Add ground_truth to the main dataframe as its own column
        if ground_truth:
            self.add_ground_truth(ground_truth_df, inplace=True)

        # TODO load images
        self.images =

        # Encode the labels and data if desired.
        self.label_encoder = None if encode_columns is None else self.encode_labels(encode_columns)

    def load_images(image_dir, bounding_box_dir=None, output=None, overwrite=False, shape=None, mask=False, by_sample=True):
        """Load the images and optionally crop  by the bounding box files.

        Parameters
        ----------
        image_dir: str
        bounding_box_dir: str
            If provided, will load the bounding box meta data and crop the
            images by their respective bounding boxes at the same time as
            loading the images. Assumes that the bounding box directory has the
            same structre as the image directory and that the bounding box info
            is contained within xml file of the same name as the image file it
            corresponds to.
        output: str
            Saves the data to the provided filepath as TFRecords
        overwrite: bool
            If True, writes to the given output directory despite it already
            existing, otherwise creates a new directory based on the current
            datetime as a child directory of the output directory.
        shape: tuple of ints
            The shape of the images to be reshaped to if provided, otherwise no
            resizing is done.
        mask: bool
            If True, creates an additional mask channel for the image based on
            the bounding box that indicates the active area of the image.
        by_sample: bool
            Returns the
        """
        if not os.path.isdir(image_dir):
            raise IOError(f'The directory `{image_dir}` does not exist.')

        class_to_img_list = {}
        for class_dir in os.listdir(image_dir)
            class_to_img_list[class_dir] = []

            class_dir_path = os.path.join(image_dir, class_dir)

            if os.path.isdir(bounding_box_dir):
                bounding_box_class_dir = os.path.join(bounding_box_dir, class_dir)
            else:
                bounding_box_class_dir = None

            for filename for os.listdir(class_dir_path):
                if bounding_box_class_dir_path:
                    bndbox = ET.parse(os.path.join(
                        bounding_box_class_dir_path,
                        filename.rpartition('.')[0],
                    )).getroot().find('object/bndbox')

                    ymin = int(bndbox.find('ymin').text)
                    ymax = int(bndbox.find('ymin').text)
                    xmin = int(bndbox.find('xmin').text)
                    xmax = int(bndbox.find('xmin').text)

                    img = cv2.imread(
                        os.path.join(class_dir_path, filename),
                    )[ymin : ymax, xmin : xmax]

                    if mask:
                        img = np.dstack((img, np.ones(img.shape)))

                    # padding, pad smaller dim on both sides, create mask layer.
                    diff = (ymax - ymin) - (xmax - xmin)
                    if diff > 0:
                        # y > x
                        img = np.pad(
                            img,
                            (
                                (0, 0),
                                (math.floor(diff / 2), math.ceil(diff / 2)),
                                (0, 0),
                            ),
                            'constant',
                        )
                    elif diff < 0:
                        # y < x
                        img = np.pad(
                            img,
                            (
                                (math.floor(diff / 2), math.ceil(diff / 2)),
                                (0, 0),
                                (0, 0),
                            ),
                            'constant',
                        )

                    if shape:
                        img = cv2.resize(
                            img,
                            shape,
                            interpolation=cv2.INTER_CUBIC,
                        )

                    class_to_img_list[class_dir].append(img)
                else:
                    class_to_img_list.append(cv2.imread(
                        os.path.join(class_dir_path, filename),
                    ))

        if not output and not by_sample:
            return class_to_img_list

        if os.path.exists(output) and not overwrite:
            date_dir = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            logging.warning(f'output `{output}` already exists and overwrite is `{overwrite}`. output changed to `{output}/{date_dir}`.')
            output = os.path.join(output, date_dir)
        os.makedirs(output, exist_ok=True)

        # TODO save the images as TFRecord
        # with tf.python_io.TFRecordWriter(output) as writer:
        #    writer.writer(().SerializeToString())

        return class_to_img_list if not by_sample else df

    def load_tfrecords():
        return

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
