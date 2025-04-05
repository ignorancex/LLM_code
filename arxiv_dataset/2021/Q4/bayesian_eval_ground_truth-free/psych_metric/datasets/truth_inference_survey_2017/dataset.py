"""Dataset class handler for truth inference survey 2017 data."""
import os

#import numpy as np
import pandas as pd

from psych_metric.datasets.base_dataset import BaseDataset

try:
    ROOT = os.environ['ROOT']
    HERE = os.path.join(ROOT, 'psych_metric/datasets/truth_inference_survey_2017/truth_inference_survey_2017_data/')
except KeyError:
    HERE = None

class TruthSurvey2017(BaseDataset):
    """class that loads and serves data from truth inference survey 2017

    Attributes
    ----------
    dataset : str
        Name of specific (sub) dataset contained within this class.
    task_type : str
        The type of learning task the dataset is intended for. This can be one
        of the following: 'regression', 'binary_classification', 'classification'
    df : pandas.DataFrame
        contains the data of the dataset in a standardized format (typically
        an annotation list where each row is an individual's annotation of one
        sample. Must contain the columns: 'worker_id' and 'worker_label'.
        'ground_truth' is also a common column name when ground truth is
        included with the original dataset. 'sample_id' will exist when no
        features are provided, or some features need loaded.
    label_set : set
        Set containing the complete original labels.
    label_encoder : {str: sklearn.preprocessing.LabelEncoder}
        Dict of column name to Label Encoder for the labels. None if no
        encodings used.
    """
    datasets = frozenset([
        'd_Duck Identification', 'd_jn-product', 'd_sentiment',
        's4_Dog data', 's4_Face Sentiment Identification', 's4_Relevance',
        's5_AdultContent', 'f201_Emotion_FULL'
    ])

    def __init__(self, dataset='d_Duck Identification', dataset_filepath=None, encode_columns=None, ground_truth=False):
        """initialize class by loading the data

        Parameters
        ----------
        dataset : str
            the name of one of the subdatasets corresponding to file name
        encode_columns : list, optional
            Encodes columns provided as list of str; dataframe uses raw values
            by default.
        """
        self.load_dataset(dataset, dataset_filepath, encode_columns)

    def load_dataset(self, dataset='d_Duck Identification', dataset_filepath=None, encode_columns=None):
        self._check_dataset(dataset, TruthSurvey2017.datasets)
        self.dataset = dataset

        if dataset_filepath is None:
            if HERE is None or 'truth_inference_survey_2017_data' not in HERE:
                raise ValueError('A path to the dataset file was not provided either by the `dataset_filepath` parameter or by the ROOT environment variable. Global variable HERE is `%s`. It is recommended to use the `dataset_filepath` parameter to provide the filepath.' % HERE)
            dataset_filepath = HERE
        self.data_dir = dataset_filepath

        # Set the dataset's expected task type
        if 'd' == self.dataset[0]:
            self.task_type = 'binary_classification'
        elif 's' == self.dataset[0]:
            self.task_type = 'classification'
            # NOTE the dog breeds identification is probably a hierarchial classifiaction problem
        elif 'f' == self.dataset[0]:
            self.task_type = 'regression'

        # Read in and save data
        annotation_file = os.path.join(dataset_filepath, self.dataset, 'answer.csv')
        self.df = pd.read_csv(annotation_file)
        # change to standardized column names. #sample_id is a feature.
        self.df.columns = ['sample_id', 'worker_id', 'worker_label']

        labels_file = os.path.join(dataset_filepath, self.dataset, 'truth.csv')
        ground_truth_df = pd.read_csv(labels_file)
        # change to standardized column names.
        ground_truth_df.columns = ['sample_id', 'worker_label']

        # Save labels set
        self.label_set = set(ground_truth_df['worker_label'].unique()) if dataset != 'f201_Emotion_FULL' else None

        # Add ground_truth to the main dataframe as its own column
        if ground_truth:
            self.add_ground_truth(ground_truth_df, inplace=True)

        if encode_columns == True:
            # The default columns to encode for each data subset.
            if dataset in {'d_jn-product', 's4_Relevance', 's5_AdultContent'}:
                encode_columns = {'sample_id', 'worker_id'}
            elif dataset in {'d_sentiment', 's4_Face Sentiment Identification', 'f201_Emotion_FULL'}:
                encode_columns = {'worker_id'}

        # Load and save the features if any.
        self.features = None

        # Encode the labels and data if desired.
        self.label_encoder = None if encode_columns is None else self.encode_labels(encode_columns)

    def convert_to_sparse_matrix(self, df):
        """Convert provided dataframe into a sparse matrix equivalent.

        Converts the given dataframe of expected format equivalent to this
        dataset's datafile structure into a sparse matrix dataframe where the
        row corresponds to each sample and the columns are structured as
        features, df of worker_id, where missing annotations are NA.

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
        # This is desirable from a standardization perspective of making the
        # data fit the format of model functions such as those in scikit learn.
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
