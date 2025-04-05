import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from psych_metric.utils import tqdm

from psych_metric.datasets.base_dataset import BaseDataset

ROOT = os.environ['ROOT']
HERE = os.path.join(ROOT, 'psych_metric/datasets/volcano/')

class SimulatedVolcanoMultinomial(BaseDataset):
    """class that generates fake volcano paper data
    """

    def __init__(self, posterior, prior, n_annos=(1,10),  N=1000):
        """initialize class by loading the data

        Parameters
        ----------
        """
        n_rows = [int(np.rint(p*N)) for p in prior]
        if sum(n_rows) != N: n_rows[-1] += (N - sum(n_rows)) # make sure it adds up to N
        ks = [np.random.randint(n_annos[0], n_annos[1]+1, (nr,)) for nr in n_rows]
        xs = [self.generate_single_class(li, ni) for li, ni in zip(posterior, ks)]

        self.n_classes = len(prior)
        self.N = N
        self.X = np.concatenate(xs, axis=0)
        self.K = np.concatenate(ks, axis=0)
        self.Y = np.zeros((N,), dtype=int)
        for i, index in enumerate(np.cumsum(n_rows)[:-1]):
            self.Y[index:] = i+1

        self.df = self.aggregate_in_df(self.X, self.K, self.Y)

    def generate_single_class(self, posterior, num_annos):
        x = np.array(
            [np.random.multinomial(k, posterior, size=()) for k in num_annos]
        )
        return x

    def aggregate_in_df(self, X, K, Y):
        d = {
                'annotations': [Xi for Xi in X],
                'n_annotators': K.tolist(),
                'label': Y.tolist(),
        }
        return pd.DataFrame.from_dict(d)

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

    def display(self, i, ylim=None, ax=None):
        d = self.__getitem__(i)
        if ax is None: fig, ax = plt.subplots(1,1, figsize=(3,3))
        c = d['annotations']
        v = range(len(c))
        ax.bar(v, c)
        if ylim is not None: ax.set_ylim(ylim)
        ax.set_xlabel('Features')
        ax.set_ylabel('Count')
        ax.set_title('True Class: {}'.format(d['label']))


class SimulatedVolcanoMultinomialTemporal(BaseDataset):
    """class that generates fake volcano paper data
    """

    def __init__(self, posterior, prior, transitions, n_annos=(1,10), n_steps=(1,10), N=1000):
        """initialize class by loading the data

        Parameters
        ----------
        """

        self.X = list()
        for row in tqdm(range(N)):
            x = self.generate_single_row(
                    posterior=posterior, prior=prior, transitions=transitions,
                    n_annos=n_annos, n_steps=n_steps
            )
            self.X.append(x)

    def generate_single_row(self, posterior, prior, transitions, n_annos, n_steps):
        num_steps = np.random.randint(n_steps[0], n_steps[1]+1, size=())
        num_annos = np.random.randint(n_annos[0], n_annos[1]+1, size=(num_steps,))

        X = list()
        current_state = np.random.choice(len(prior), size=(), p=prior)
        for i, k in enumerate(num_annos):
            x = np.random.multinomial(k, pvals=posterior[current_state], size=())
            X.append(x)
            current_state = np.random.choice(len(prior), size=(), p=transitions[current_state])

        return np.stack(X, axis=0)

    def __len__(self):
        """ get size of dataset

        Returns
        -------
        int
            number of annotations in dataset
        """
        return len(self.X)

    def __getitem__(self, i):
        """ get specific row from dataset

        Returns
        -------
        dict:
            {header: value, header: value, ...}
        """
        return self.X[i]

    def display(self, i, ylim=None):
        X = self.__getitem__(i)
        n_steps = len(X)
        fig, axarr = plt.subplots(1, n_steps, figsize=(3*n_steps, 3))
        v = range(len(X[0]))
        for t in range(n_steps):
            c = X[t]
            ax = axarr[t]
            ax.bar(v, c)
            if ylim is not None: ax.set_ylim(ylim)
            ax.set_xlabel('Features')
            if t == 0: ax.set_ylabel('Count')
            ax.set_title('$t = {}$'.format(t))
