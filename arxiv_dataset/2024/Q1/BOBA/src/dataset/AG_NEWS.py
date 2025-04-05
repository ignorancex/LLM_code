import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from nltk.tokenize import word_tokenize


class AG_NEWS(Dataset):
    """
    AG-News Dataset.
    Preprocess to tokens during loading
    """

    def __init__(self, root, max_len=70, vocab=None, train=True, download=True):
        """
        :param root: root of the dataset
        :param max_len: clip the sentence to this length if the sentence is longer
        :param vocab: function: convert tokens to indices
        :param train: load the training set / testing set
        :param download:
        """

        self.train = train
        self.max_len = max_len
        self.vocab = vocab

        self.root = os.path.join(root, 'ag_news')
        if self.train:
            self.path = os.path.join(root, 'ag_news/train.csv')
        else:
            self.path = os.path.join(root, 'ag_news/test.csv')

        if download:
            self.download()

        self.data, self.lens, self.targets = self.load_and_preprocess()

        self.classes = [
            '0 - World',
            '1 - Sports',
            '2 - Business',
            '3 - Sci/Tech',
        ]

    def download(self):

        if not os.path.exists(self.path):

            if not os.path.exists(self.root):
                os.mkdir(self.root)

            from urllib.request import urlretrieve

            if self.train:
                url = 'https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/train.csv'
            else:
                url = 'https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/test.csv'

            print('Downloading AG_NEWS dataset ...')
            urlretrieve(url, self.path)

    def load_and_preprocess(self):
        df = pd.read_csv(self.path, header=None)

        print('Preprocessing AG_NEWS dataset ...')
        # preprocessing X
        # iterate every line
        # 1. join title and description
        # 2. replace '\\' with ' ' because '\\' cannot be automatically handled by the tokenizer
        # 3. convert to lower case, because GloVe has only lower case words
        # 4. tokenize
        # 5. convert to indices
        # 6. padding or clipping

        X = []
        lens = []
        for i in range(len(df)):

            sentence = ' '.join([df[1][i], df[2][i]]).replace('\\', ' ').lower()  # 1, 2, 3
            tokens = word_tokenize(sentence)  # 4
            word_indices = [self.vocab(token) for token in tokens]  # 5

            # 6
            sent_len = len(word_indices)
            if sent_len < self.max_len:  # do padding with 0
                word_indices = word_indices + [0] * (self.max_len - sent_len)
                lens.append(sent_len)
            else:  # do clipping
                word_indices = word_indices[:self.max_len]
                lens.append(self.max_len)

            X.append(word_indices)

        X = torch.LongTensor(X)
        lens = torch.LongTensor(lens)
        Y = torch.LongTensor(df[0]) - 1  # convert 1 - 4 to 0 - 3

        return X, lens, Y

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item], self.lens[item], self.targets[item]


if __name__ == '__main__':
    from torch.utils.data import DataLoader

    news = AG_NEWS(root='../../../data', vocab=lambda x: 0, train=False)
    print(len(news[0][0]), len(news[1][0]), len(news[2][0]), len(news[3][0]))
    loader = DataLoader(news, batch_size=4, shuffle=False)
    for i, (X, lens, Y) in enumerate(loader):
        print(X.shape, lens.shape, Y.shape)
        print(lens)
        break
