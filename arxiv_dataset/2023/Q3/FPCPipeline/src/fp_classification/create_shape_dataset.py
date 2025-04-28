"""

Using a pre-computed train/test dataset, drop all non-shape features (avoid recomputation)

@author: Ishaan Bhat
@email: ishaan@isi.uu.nl

"""
import os
from argparse import ArgumentParser
import pandas as pd
import numpy as np

def filter_dataframe(df):

    shape_df = df[['Patient ID',  'MeshVolume', 'VoxelVolume',
                     'SurfaceArea', 'SurfaceVolumeRatio', 'Sphericity',
                     'Maximum3DDiameter', 'Maximum2DDiameterSlice', 'Maximum2DDiameterColumn',
                     'Maximum2DDiameterRow', 'MajorAxisLength', 'MinorAxisLength',
                     'LeastAxisLength', 'Elongation', 'Flatness', 'label', 'slice']]
    return shape_df

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, default=None)
    parser.add_argument('--dataset', type=str, default='lits')
    parser.add_argument('--out_dir', type=str, default=None)

    args = parser.parse_args()

    if os.path.exists(args.out_dir) is False:
        os.makedirs(args.out_dir)

    train_dataset = filter_dataframe(df=pd.read_pickle(os.path.join(args.dataset_dir, 'train_dataset.pkl')))
    test_dataset = filter_dataframe(df=pd.read_pickle(os.path.join(args.dataset_dir, 'test_dataset.pkl')))

    train_dataset.to_pickle(os.path.join(args.out_dir, 'train_dataset.pkl'))
    test_dataset.to_pickle(os.path.join(args.out_dir, 'test_dataset.pkl'))

