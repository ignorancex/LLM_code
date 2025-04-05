"""Dataset class handler for Ipeirotis2010 data."""
import os
import csv

import pandas as pd
from sklearn.preprocessing import LabelEncoder

from psych_metric.datasets.base_dataset import BaseDataset

try:
    ROOT = os.environ['ROOT']
    HERE = os.path.join(ROOT, 'psych_metric/datasets/ipeirotis_2010/ipeirotis_2010_data/')
except KeyError:
    HERE = None

class Ipeirotis2010(BaseDataset):
    """class that loads and serves data from Ipeirotis 2010

    Attributes
    ----------
    dataset : str
        Name of specific dataset
    df : pandas.DataFrame
        Contains the data of the dataset in a standardized format. Typically
        that format is an annotation list where each row is an individual's
        annotation of one sample. Must contain the columns: 'worker_id' and
        'worker_label'.  'ground_truth' is also a common column name when ground
        truth is included with the original dataset. 'sample_id' will exist when
        no features are provided, or some features need loaded.
    label_set : set
        Set containing the complete original labels
    label_encoder : {str: sklearn.preprocessing.LabelEncoder}
        Dict of column name to Label Encoder for the labels. None if no
        encodings used.
    """
    datasets = frozenset([
        'AdultContent', 'AdultContent2', 'AdultContent3',
        'BarzanMozafari', 'CopyrightInfringement',
        'HITspam-UsingCrowdflower', 'HITspam-UsingMTurk',
        'JeroenVuurens'
    ])

    def __init__(self, dataset='AdultContent', dataset_filepath=None, encode_columns=None, ground_truth=False):
        """initialize class by loading the data

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

        self.load_dataset(dataset, dataset_filepath, encode_columns, ground_truth)

    def load_dataset(self, dataset='AdultContent', dataset_filepath=None, encode_columns=None, ground_truth=False):
        self._check_dataset(dataset, Ipeirotis2010.datasets)

        if dataset_filepath is None:
            if HERE is None or 'ipeirotis_2010_data' not in HERE:
                raise ValueError('A path to the dataset file was not provided either by the `dataset_filepath` parameter or by the ROOT environment variable. Global variable HERE is `%s`. It is recommended to use the `dataset_filepath` parameter to provide the filepath.' % HERE)
            dataset_filepath = HERE
        self.data_dir = dataset_filepath

        # TODO perhaps separate these conditionals into data specific load functions. It'll be much cleaner.
        self.dataset = 'AdultContent3-HCOMP2010' if dataset == 'AdultContent3' else dataset

        # Set the task type of the dataset
        self.task_type = 'binary_classification' if self.dataset in {'BarzanMozafari', 'HITspam-UsingCrowdflower', 'HITspam-UsingMTurk', 'JeroenVuurens'} else 'classification'

        # Read in and save data
        if dataset == 'JeroenVuurens':
            annotation_file = os.path.join(dataset_filepath, self.dataset, 'votes')
            self.df = pd.read_csv(annotation_file, names=['time', 'sample_id', 'worker_id', 'worker_label'], delimiter='\t')
        else:
            annotation_file = os.path.join(dataset_filepath, self.dataset, 'labels.txt')
            self.df = pd.read_csv(annotation_file, names=['worker_id', 'sample_id', 'worker_label'], delimiter='\t')
            # NOTE the sample_id is a url
            # reorder columns into format of features, worker_id, label
            self.df = self.df[['sample_id', 'worker_id', 'worker_label']]

        # Read in ground truth, if any.
        if ground_truth:
            if ('Adult' in dataset and '3' not in dataset) or dataset == 'CopyrightInfringement' or 'HITspam-Using' in dataset:
                ground_truth_file = os.path.join(dataset_filepath, self.dataset, 'gold.txt')
                ground_truth_dict = self.read_csv_to_dict(ground_truth_file, sep='\t')
                self.add_ground_truth(ground_truth_dict, inplace=True)

            elif dataset == 'BarzanMozafari':
                ground_truth_file = os.path.join(dataset_filepath, self.dataset, 'evaluation.txt')
                ground_truth_dict = self.read_csv_to_dict(ground_truth_file, sep='\t')
                self.add_ground_truth(ground_truth_dict, inplace=True)


        # Read in and save the expected label set, or infer the labels from data
        if 'HITspam-UsingCrowdflower' == dataset:
            self.label_set = frozenset({'Yes', 'No'})
        elif 'HITspam-UsingMTurk' == dataset:
            self.label_set = frozenset({'YES', 'NO'})
        elif dataset == 'JeroenVuurens' or dataset == 'BarzanMozafari':
            self.label_set = frozenset({0, 1})
        else:
            labels_file = os.path.join(dataset_filepath, self.dataset, 'categories.txt')
            self.label_set = self._load_labels(labels_file)

        # Encode the labels and data if desired
        if 'AdultContent' in dataset or dataset == 'CopyrightInfringement' or 'HITspam' in dataset:
            if encode_columns == True:
                encode_columns = {'sample_id', 'worker_id', 'worker_label'}
            column_labels = {'worker_label': self.label_set}
        elif dataset == 'BarzanMozafari':
            if encode_columns == True:
                encode_columns = {'worker_id'}
            column_labels = None
        elif dataset == 'JeroenVuurens':
            if encode_columns == True:
                encode_columns = {'sample_id'}
            column_labels = None
        else:
            column_labels = None

        self.label_encoder =  None if encode_columns is None else self.encode_labels(encode_columns, column_labels)

    def _load_labels(self, csv_path=None):
        """ Load labels from the published dataset categories file, or infer
        from the current dataframe.

        Parameters
        ----------
        csv_path : str, optinal
            path of categories file to load labels from; infer from data if None

        Returns
        -------
        frozenset
            frozenset of the categories as the label set.
        """
        if csv_path is None:
            # infer labels from data, will exclude any missing labels
            return frozenset(self.df['worker_label'].unique())

        # read in the labels provided
        label_set = set()
        with open(csv_path, 'r') as csvfile:
            csv_reader = csv.reader(csvfile, delimiter='\t')
            for row in csv_reader:
                label_set.add(row[0])
        return frozenset(label_set)

    def convert_to_sparse_matrix(self, df):
        """Convert provided dataframe into a sparse matrix equivalent.

        Converts the given dataframe of expected format equivalent to this
        dataset's datafile structure into a sparse matrix dataframe where the
        row corresponds to each sample and the columns are structured as
        features, annotations of worker_id, where missing annotations are NA.

        Parameters
        ----------
        annotations : pandas.DataFrame
            Dataframe to be converted into a sparse matrix format.

        Returns
        -------
        pandas.DataFrame
            Data Frame of annotations in a sparse matrix format.

        """
        #TODO decide if this is desireable, then implement if it is desireable.
        raise NotImplementedError

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
