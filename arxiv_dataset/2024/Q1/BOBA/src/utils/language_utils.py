import torch
import os


class GloVe:
    """
    GloVe word embedding (used in AG-News experiments)
    """

    def __init__(self, root='../data', name='6B', dim=50, topk=None, download=True):

        self.name = name
        self.dim = dim
        self.topk = topk

        self.root = os.path.join(root, 'glove')
        self.file_name = 'glove.%s.%dd.txt' % (name, dim)
        self.path = os.path.join(root, 'glove', self.file_name)

        if download:
            self.download()

        self.load_and_preprocess()

    def download(self):
        if not os.path.exists(self.path):
            if not os.path.exists(self.root):
                os.makedirs(self.root)

            from urllib.request import urlretrieve
            import zipfile

            urls = {
                '42B': 'http://nlp.stanford.edu/data/glove.42B.300d.zip',
                '840B': 'http://nlp.stanford.edu/data/glove.840B.300d.zip',
                'twitter.27B': 'http://nlp.stanford.edu/data/glove.twitter.27B.zip',
                '6B': 'http://nlp.stanford.edu/data/glove.6B.zip',
            }

            url = urls[self.name]
            self.zip_path = os.path.join(self.root, url.split('/')[-1])

            if not os.path.exists(self.zip_path):
                print('Downloading GloVe word embeddings')
                urlretrieve(url, self.zip_path)

            # upzip file
            print('Unzipping GloVe word embeddings')
            with zipfile.ZipFile(self.zip_path) as z:
                z.extractall(self.root)

            # zfile = zipfile.ZipFile(self.zip_path, 'r')
            #
            # # for filename in zfile.namelist():
            # #     print(filename)
            # data = zfile.read(self.file_name)
            # with open(self.path, 'wb') as f:
            #     f.write(data)

    def load_and_preprocess(self):
        with open(self.path, 'r') as f:
            lines = f.readlines()

        # keep only the most frequent words
        if self.topk is not None:
            lines = lines[:self.topk]

        lines = [line.split() for line in lines]

        self.vocab = [l[0] for l in lines]
        self.vocab2ind = {word: i for i, word in enumerate(self.vocab)}

        emb_floats = [[float(n) for n in l[1:]] for l in lines]
        emb_floats.insert(0, [0.0 for _ in range(self.dim)])  # for unknown word and padding

        self.weights = torch.Tensor(emb_floats)

    def stoi(self, word):
        if word in self.vocab2ind:
            return self.vocab2ind[word] + 1
        else:
            return 0

    def itos(self, idx):
        if idx == 0:
            return '<UNK>'
        else:
            return self.vocab[idx - 1]  # 0 is for padding
