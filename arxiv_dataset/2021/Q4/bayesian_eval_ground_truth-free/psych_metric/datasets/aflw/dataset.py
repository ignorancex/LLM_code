import ast
import os

import cv2
import imageio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from psych_metric.datasets.base_dataset import BaseDataset
import psych_metric.utils as utils

try:
    ROOT = os.environ['ROOT']
    HERE = os.path.join(ROOT, 'psych_metric/datasets/aflw/')
except KeyError:
    HERE = None

class FirstImpressions(BaseDataset):
    """Dataset wrapper class for the First Impression dataset

    Attributes
    ----------
    dataset : str
        Name of specific (sub) dataset contained within this class.
    task_type : str
        The type of learning task the dataset is intended for. This can be one
        of the following: 'regression', 'binary_classification', 'classification'
    df : pandas.DataFrame
        contains the data of the dataset in a standardized format, typically
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
    datasets = frozenset(['age', 'dominance', 'iq', 'trustworthiness'])

    def __init__(self, dataset='age', dataset_filepath=None, encode_columns=None):
        self._check_dataset(dataset, FirstImpressions.datasets)
        self.dataset = dataset

        # Set dataset's expected task type
        # NOTE this could be percieved as a regression task, or at least an ordinal task
        self.task_type = 'regression' # regression or ordinal due to ints.

        if dataset_filepath is None:
            if HERE is None or 'first_impressions' not in HERE:
                raise ValueError('A path to the dataset file was not provided either by the `dataset_filepath` parameter or by the ROOT environment variable. Global variable HERE is `%s`. It is recommended to use the `dataset_filepath` parameter to provide the filepath.' % HERE)
            dataset_filepath = os.path.join(HERE, 'first_impressions_data')

        # Read in and format the data as an annotation list.
        annotation_file = os.path.join(dataset_filepath, self.dataset + '.csv')
        self.df = pd.read_csv(annotation_file)
        # TODO worker_label is either rating or response.
        self.df.columns = ['random_index_artifact', 'sample_id', 'worker_id', 'duration', 'start_time', 'reaction_time', 'src', 'type', 'ground_truth', 'worker_label', 'norm_response', 'diff']

        # Save label set TODO make label set the range of values for regression tasks
        self.label_set = None

        if encode_columns == True:
            # The default columns to encode for each data subset.
            encode_columns = {'src', 'type'}

        # label encoder can be used for encoding the
        self.label_encoder = None

        # TODO merge this with the below classes, such that all functionality is perserved for each use case.

class FirstImpressionsSparse(BaseDataset):
    def __init__(self, dataset='trustworthiness'):
        self.dataset = dataset
        self.df_path = os.path.join(HERE, 'first_impressions_data', dataset + '.csv')
        self.load_csv()

    def load_csv(self):
        self.df = pd.read_csv(self.df_path)


class FirstImpressionsDense(BaseDataset):
    def __init__(self, stage='train', augment=False, numpy=False):
        self.numpy = numpy
        self.augment = augment
        self.stage = stage
        self.annotations = os.path.join(HERE, 'first_impressions_data', 'aggregated.csv')
        if self.numpy:
            self.img_dir = os.path.join(HERE, 'aflw-att-numpy')
        else:
            self.img_dir = os.path.join(HERE, 'aflw-att-images')
        self.df = self.load_csv()
        self.set_multinomials()

    @staticmethod
    def get_hist(votes):
        if isinstance(votes, str):
            votes = ast.literal_eval(votes)
        bins = range(1, 9)
        return np.histogram(votes, bins=bins)[0]

    def set_multinomials(self):
        cd = {
                'Trustworthiness_raw': 'Trustworthiness_hist',
                'Dominance_raw': 'Dominance_hist',
                'Age_raw': 'Age_hist',
                'IQ_raw': 'IQ_hist',
        }
        for f, t in cd.items():
            self.df[t] = self.df[f].map(self.get_hist)

    def get_multinomial_arrays(self):
        arrays = dict()
        for trait in ['Trustworthiness', 'Dominance', 'IQ', 'Age']:
            col = trait + '_hist'
            x = np.stack(self.df[col], axis=0)
            arrays[trait] = x
        return arrays

    def jpg_to_numpy(self, f):
        return f[:-len('jpg')] + 'npy'

    def load_csv(self):
        df = pd.read_csv(self.annotations, index_col=0, header=0)
        df = df[df['file_id'].map(lambda x: isinstance(x, str))]
        df = df[
            (df['Trustworthiness_num'] != 0) & \
            (df['Dominance_num'] != 0) & \
            (df['Age_num'] != 0) & \
            (df['IQ_num'] != 0)
        ]
        if self.numpy:
            df['file_id'] = df['file_id'].map(self.jpg_to_numpy)
        df = df[df['split'] == self.stage]

        return df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        i = i % len(self.df)
        row = self.df.iloc[i]
        imf = os.path.join(self.img_dir, row['file_id'])

        if self.numpy:
            img = np.load(imf)
        else:
            img = imageio.imread(imf)

        # cut out the image with jitter at train time
        box = int(row['x']), int(row['y']), int(row['w']), int(row['h'])
        if self.augment: box = utils.jitter_box(box, pw=0.2, ph=0.2)
        box = utils.make_box_square(box)
        roi = utils.cut_out_box(img, box, pad_mode='edge')
        roi = cv2.resize(roi, (224,224))

        # return trustworthiness, dominance, age, iq
        age = row['Age_raw']
        trust = row['Trustworthiness_raw']
        dom = row['Dominance_raw']
        iq = row['IQ_raw']

        raw_to_mean = lambda x: (np.mean(ast.literal_eval(x)) - 1.) / 6.0
        for arr in [age, trust, dom, iq]:
            arr = ast.literal_eval(arr)
            if len(arr) == 0:
                print(i)

        ret = {
            'Trustworthiness': raw_to_mean(trust),
            'Dominance': raw_to_mean(dom),
            'Age': raw_to_mean(age),
            'IQ': raw_to_mean(iq),
            'image': roi,
        }

        return ret

    def display(self, i, ax=None):
        r = self[i]
        if ax is None: fig, ax = plt.subplots(1,1, figsize=(3,3))
        title = '\n'.join([k + ": {:.2f}".format(v) for k, v in r.items() if k != 'image'])
        ax.imshow(r['image'])
        ax.set_title(title)
