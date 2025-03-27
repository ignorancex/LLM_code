import argparse
import glob
import pickle

import h5py
import numpy as np
from rich import print
from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--files', default='data/sokoban/dagger/unfiltered/*.data')
    args = parser.parse_args()

    files = glob.glob(args.files)
    for file in tqdm(files):
        with open(file, 'rb') as f:
            data = pickle.load(f)

        hdf5_file = h5py.File(file.replace('.data', '.h5'), mode='w')

        hdf5_file.create_dataset('states', data=np.asarray(data['states']))
        hdf5_file.create_dataset('actions', data=np.asarray(data['actions']))
        hdf5_file.create_dataset('values', data=np.asarray(data['q_values' if 'q_values' in data else 'values']))

        hdf5_file.close()

        print('Total samples: ', len(data['actions']))
