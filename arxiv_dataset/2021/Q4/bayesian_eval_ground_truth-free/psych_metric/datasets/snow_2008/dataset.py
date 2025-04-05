import os
import pandas as pd

from psych_metric.datasets.base_dataset import BaseDataset

try:
    ROOT = os.environ['ROOT']
    HERE = os.path.join(ROOT, 'psych_metric/datasets/snow_2008/')
except KeyError:
    HERE = None

class Snow2008(BaseDataset):
    """class that loads and serves data from Snow 2008

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
        Set containing the complete original labels.
    """
    datasets = frozenset([
        'anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise',
        'valence', 'rte', 'temp',  'wordsim', 'wsd'
    ])

    def __init__(self, dataset='anger', dataset_filepath=None):
        """initialize class by loading the data. Ground truth is always loaded.

        Parameters
        ----------
        dataset : str
            the name of one of the subdatasets corresponding to file name
        """
        self._check_dataset(dataset, Snow2008.datasets)
        self.dataset = dataset

        if dataset == 'wsd':
            print('`wsd`: word sense disambiguation is either a mapping or hierarchial classifiaction problem, eitherway, none of the truth inference models will handle this correctly, as far as is known at the moment.')
            raise NotImplementedError

        if self.dataset in {'anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'valence'}:
            self.task_type = 'regression'
            # NOTE all emotion and valence could be classifiaction with 100 or 200 labels (integers), which I suppose is really ordering, rather than classification.
            self.label_set = None
        elif self.dataset == 'wordsim':
            self.task_type = 'regression'
            self.label_set = None
        elif self.dataset == 'rte':
            self.task_type = 'binary_classification'
            self.label_set = frozenset({0, 1})
        elif self.dataset == 'temp':
            # NOTE temp=temporal is ordered labels 'strictly before' and 'stritly after'
            self.task_type = 'binary_classification'
            self.label_set = frozenset({1, 2})
        else: # wsd
            self.task_type = 'mapping'
            # NOTE wsd: word sense disambiguation is a mapping problem, not a classifiaction problem.
            # The labels will result in misleading class relationships that are non-existent.
            # Furthermore, this is a difficult mapping problem where the item and its possible things to be mapped to both change, rather than keeping a static target to map to.
            # Perhaps, this could be viewed as some form of hierarchial classification.

        if dataset_filepath is None:
            if HERE is None or 'snow_2008' not in HERE:
                raise ValueError('A path to the dataset file was not provided either by the `dataset_filepath` parameter or by the ROOT environment variable. Global variable HERE is `%s`. It is recommended to use the `dataset_filepath` parameter to provide the filepath.' % HERE)
            dataset_filepath = os.path.join(HERE, 'snow_2008_data')
        self.data_dir = dataset_filepath

        annotation_file = '{}.standardized.tsv'.format(self.dataset)
        annotation_file = os.path.join(dataset_filepath, annotation_file)
        self.df = self.load_tsv(annotation_file)

    def load_tsv(self, f):
        """ Read and parse the published dataset file and set column names to
        the standardized annotation list format.

        Parameters
        ----------
        f : str
            path of tsv file

        Returns
        -------
        pandas.DataFrame
            Data Frame of annotations

        """
        df = pd.read_csv(f, header=0, delimiter='\t')
        df.columns = ['amt_annotation_ids', 'worker_id', 'sample_id', 'worker_label', 'ground_truth']
        # NOTE keeping gold for now, uncertain if only ground_truth will be for those with ACTUAL ground truth able to be determined.
        return df

    def __len__(self):
        """ get size of dataset

        Returns
        -------
        int
            number of annotations in dataset
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
