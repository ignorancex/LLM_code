import pandas as pd
import logging
import os
import math

from config import FACEBOOK_DATASET_TRAIN, FACEBOOK_DATASET_TEST, FACEBOOK_DATASET_DEV, \
    FACEBOOK_DATASET, FACEBOOK_DATASET_DIR, CSFD_DATASET_TRAIN, CSFD_DATASET_TEST, CSFD_DATASET_DEV, CSFD_DATASET, \
    CSFD_DATASET_DIR, MALL_DATASET_DIR, MALL_DATASET, MALL_DATASET_DEV, MALL_DATASET_TEST, MALL_DATASET_TRAIN, \
    RANDOM_SEED, IMDB_DATASET_CL_DIR, IMDB_DATASET_CL_TRAIN, IMDB_DATASET_CL_TEST, \
    IMDB_DATASET_CL, IMDB_DATASET_DIR, IMDB_DATASET_TEST, IMDB_DATASET_TRAIN, IMDB_DATASET

logger = logging.getLogger(__name__)

class DatasetLoader(object):

    def __init__(self, maxt_train_data, binary):
        self.binary = binary
        self.max_train_data = maxt_train_data

        self.train_data = None
        self.test_data = None
        self.dev_data = None
        self.datasets = []

    def get_dev_data(self):
        if self.dev_data is None:
            self.load_data()
        return self.dev_data

    def get_test_data(self):
        if self.test_data is None:
            self.load_data()
        return self.test_data

    def get_cutted_train_data(self, data_df):

        data_to_return = data_df
        total_size = len(data_df)
        logger.info("The size of training dataset is:" + str(total_size))
        logger.info("Applying cutting to train data with value:" + str(self.max_train_data))

        new_size = -1
        if self.max_train_data <= 0:
            logger.info("No cutting is performed")
            pass
        elif 0 < self.max_train_data <= 1:
            logger.info("Cutting in percentages")
            new_size = total_size * self.max_train_data
            new_size = math.ceil(new_size)
        elif self.max_train_data > 1:
            logger.info("Cutting in absolute numbers")
            new_size = self.max_train_data
            new_size = math.ceil(new_size)
        else:
            raise Exception("Unkonwn value for max_train_data, the value:" + str(self.max_train_data))

        logger.info("New size is:" + str(new_size))
        if new_size > 1:
            data_to_return = data_to_return.head(new_size)
            data_to_return.reset_index(drop=True, inplace=True)

        return data_to_return

    def get_train_data(self):
        if self.train_data is None:
            self.load_data()

        data_to_return = self.get_cutted_train_data(self.train_data)

        return data_to_return

    def get_train_dev_data(self):
        if self.train_data is None:
            self.load_data()
        if self.dev_data is None:
            self.load_data()

        df = pd.concat([self.train_data, self.dev_data], axis=0)
        df.reset_index(drop=True, inplace=True)

        data_to_return = self.get_cutted_train_data(df)

        return data_to_return

    def load_entire_dataset(self):
        raise NotImplementedError()

    def load_data(self):
        """Loads all data"""
        raise NotImplementedError()

    def get_dataset_dir(self):
        """Returns dir of the dataset"""
        raise NotImplementedError()

    def get_class_num(self):
        """Returns number of classis"""
        if self.binary is True:
            return 2
        else:
            return 3

    def get_classes(self):
        """Returns possible clases as numbers"""
        if self.binary is True:
            return [0, 1]
        else:
            return [0, 1, 2]

    def get_class_names(self):
        """
        Returns possible names of classes
        Indices corresponds to the ones returned by method get_classes
        In default it returns three classes, can be overiden
        """
        if self.binary is True:
            return ['negative', 'positive']
        else:
            return ['negative', 'positive', 'neutral']

    def get_text4label(self, label):
        """
        Returns text for numerical label
        :param label: numerical label
        :return:  textual label 0 - negative, 1 - positive, 2 - neutral
        """
        ret = ''
        if label == 0:
            ret = 'negative'
        elif label == 1:
            ret = 'positive'
        elif label == 2:
            ret = 'neutral'
        else:
            raise Exception("Unkonw label:" + str(label))

        return ret

    def get_label4text(self, text_label):
        """
        Returns numerical label for text label
        :param text_label: text label
        :return: negative - 0, positive - 1, neutral - 2
        """
        ret = ''
        if text_label == 'negative':
            ret = 0
        elif text_label == 'positive':
            ret = 1
        elif text_label == 'neutral':
            ret = 2
        else:
            raise Exception("Unkonw text label:" + str(text_label))

        return ret

    def filter_neutral(self, df):
        if self.binary:
            df = df[df.label != 2]
            df.reset_index(drop=True, inplace=True)
        return df



class CzechFBDatasetLoader(DatasetLoader):
    def __init__(self, max_train_data, binary=False):
        super().__init__(max_train_data, binary)

    def load_data(self):
        self.train_data = pd.read_csv(os.path.join(FACEBOOK_DATASET_TRAIN))
        self.train_data = self.filter_neutral(self.train_data)

        self.test_data = pd.read_csv(os.path.join(FACEBOOK_DATASET_TEST))
        self.test_data = self.filter_neutral(self.test_data)

        self.dev_data = pd.read_csv(os.path.join(FACEBOOK_DATASET_DEV))
        self.dev_data = self.filter_neutral(self.dev_data)

    def load_entire_dataset(self):
        df = pd.read_csv(os.path.join(FACEBOOK_DATASET))
        df = self.filter_neutral(df)
        return df

    def get_dataset_dir(self):
        return FACEBOOK_DATASET_DIR


class CzechCSFDDatasetLoader(DatasetLoader):
    def __init__(self, max_train_data, binary=False):
        super().__init__(max_train_data, binary)

    def load_data(self):
        self.train_data = pd.read_csv(os.path.join(CSFD_DATASET_TRAIN))
        self.train_data = self.filter_neutral(self.train_data)

        self.test_data = pd.read_csv(os.path.join(CSFD_DATASET_TEST))
        self.test_data = self.filter_neutral(self.test_data)

        self.dev_data = pd.read_csv(os.path.join(CSFD_DATASET_DEV))
        self.dev_data = self.filter_neutral(self.dev_data)

    def load_entire_dataset(self):
        df = pd.read_csv(os.path.join(CSFD_DATASET))
        df = self.filter_neutral(df)
        return df

    def get_dataset_dir(self):
        return CSFD_DATASET_DIR

class CzechMALLCZDatasetLoader(DatasetLoader):
    def __init__(self, max_train_data, binary=False):
        super().__init__(max_train_data, binary)

    def load_data(self):
        self.train_data = pd.read_csv(os.path.join(MALL_DATASET_TRAIN))
        self.train_data = self.filter_neutral(self.train_data)

        self.test_data = pd.read_csv(os.path.join(MALL_DATASET_TEST))
        self.test_data = self.filter_neutral(self.test_data)

        self.dev_data = pd.read_csv(os.path.join(MALL_DATASET_DEV))
        self.dev_data = self.filter_neutral(self.dev_data)

    def load_entire_dataset(self):
        df = pd.read_csv(os.path.join(MALL_DATASET))
        df = self.filter_neutral(df)
        return df

    def get_dataset_dir(self):
        return MALL_DATASET_DIR




class CrossLingualCsfdImdbDataset(DatasetLoader):
    """
    It returns CSFD Czech train and test dataset as train data
    As dev data it returns CSFD Czech dev data
    As test data it returns IMDB original test data
    As special dev data it returns IMDB original train data
    """

    def __init__(self, max_train_data, binary=True):
        if binary == False:
            raise Exception("Imdb dataset is a only binary dataset")
        super().__init__(max_train_data, binary)

    def get_dataset_dir(self):
        return IMDB_DATASET_DIR

    def load_data(self):
        tmp_train_data_csfd = pd.read_csv(os.path.join(CSFD_DATASET_TRAIN))
        tmp_train_data_csfd = self.filter_neutral(tmp_train_data_csfd)

        tmp_test_data_csfd = pd.read_csv(os.path.join(CSFD_DATASET_TEST))
        tmp_test_data_csfd = self.filter_neutral(tmp_test_data_csfd)

        self.train_data = pd.concat([tmp_train_data_csfd, tmp_test_data_csfd], axis=0)
        self.train_data.reset_index(drop=True, inplace=True)

        self.dev_data = pd.read_csv(os.path.join(CSFD_DATASET_DEV))
        self.dev_data = self.filter_neutral(self.dev_data)

        self.test_data = pd.read_csv(os.path.join(IMDB_DATASET_TEST))
        self.test_data = self.filter_neutral(self.test_data)

        self.dev_data_eng = pd.read_csv(os.path.join(IMDB_DATASET_TRAIN))
        self.dev_data_eng = self.filter_neutral(self.dev_data_eng)

    def get_dev_data_eng(self):
        if self.dev_data_eng is None:
            self.load_data()
        return self.dev_data_eng

    def load_entire_dataset(self):
        df = pd.read_csv(os.path.join(IMDB_DATASET))
        df = self.filter_neutral(df)
        return df


class CrossLingualImdbCSFDDataset(DatasetLoader):
    """
     This dataset use different split than the original one

      It returns IMDB English data as train data
      As dev data it returns IMDB English test data (these are different from the original one)
      As test data it returns CSFD Czech test data
      There is additional function that returns czech CSFD Dev Data

      The entire dataset is loaded as IMDB English dataset
      """

    def __init__(self, max_train_data, binary=True):
        if binary == False:
            raise Exception("Imdb dataset is only binary dataset")
        super().__init__(max_train_data, binary)

    def get_dataset_dir(self):
        return IMDB_DATASET_CL_DIR

    def load_data(self):
        self.train_data = pd.read_csv(os.path.join(IMDB_DATASET_CL_TRAIN))
        self.train_data = self.filter_neutral(self.train_data)

        # it is correct we return test data for english as dev here
        self.dev_data = pd.read_csv(os.path.join(IMDB_DATASET_CL_TEST))
        self.dev_data = self.filter_neutral(self.dev_data)

        self.test_data = pd.read_csv(os.path.join(CSFD_DATASET_TEST))
        self.test_data = self.filter_neutral(self.test_data)

        self.dev_data_czech = pd.read_csv(os.path.join(CSFD_DATASET_DEV))
        self.dev_data_czech = self.filter_neutral(self.dev_data_czech)

    def get_dev_data_czech(self):
        if self.dev_data_czech is None:
            self.load_data()
        return self.dev_data_czech

    def load_entire_dataset(self):
        df = pd.read_csv(os.path.join(IMDB_DATASET_CL))
        df = self.filter_neutral(df)
        return df


DATASET_LOADERS = {
    "fb": CzechFBDatasetLoader,
    "csfd" : CzechCSFDDatasetLoader,
    "mallcz" : CzechMALLCZDatasetLoader,
    "imdb-csfd" : CrossLingualImdbCSFDDataset,
    "csfd-imdb" : CrossLingualCsfdImdbDataset
}
