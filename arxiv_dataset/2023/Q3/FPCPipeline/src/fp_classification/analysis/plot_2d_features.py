"""

Script to visualize 2-D features

@author: Ishaan Bhat
@email: ishaan@isi.uu.nl

"""

import matplotlib.pyplot as plt
import matplotlib as mpl
# Set style and font
plt.style.use('seaborn-paper')
mpl.rc("font", family="Serif")

import numpy as np
import os
from helper_functions import *
from argparse import ArgumentParser
import joblib
import pandas as pd

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--out_dir', type=str, required=True)

    args = parser.parse_args()

    # Read the train features DF
    train_df = pd.read_pickle(os.path.join(args.out_dir, 'train_dataset.pkl'))

    selected_features = joblib.load(os.path.join(args.out_dir, 'selected_features.pkl'))

    features_matrix, labels, _ = create_features_matrix(train_df,
                                                        selected_features)

    tp_features = features_matrix[labels == 0 , :]
    fp_features = features_matrix[labels == 1, :]

    fig, ax = plt.subplots()
    ax.scatter(tp_features[:, 0], tp_features[:, 1], c='tab:green', label='True Positive')
    ax.scatter(fp_features[:, 0], fp_features[:, 1], c='tab:orange', label='False Positive')
    ax.legend(loc='best')
    ax.set_xlabel(selected_features[0])
    ax.set_ylabel(selected_features[1])

    plt.savefig(os.path.join(args.out_dir, 'features_2d.pdf'))


