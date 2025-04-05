"""
Author: Renato Durrer
Created: 05.04.2019

File used to augment the data stored in the folder data/marked
Output to folder data/augmented
"""
import os
import numpy as np
import pandas as pd
import imgaug as ia
from imgaug import augmenters as iaa
from src.utils.funcs import x_y_derivator, scaler
import matplotlib.pyplot as plt
from matplotlib import colors
import time


class Augmenter:
    """
    Class that augments complete maps / charge stability diagrams.

    Workflow Example
    ----------------
    A = Augmenter(input_folder, output_folder, perform_augment=False, swap_axes=True)
    while(A.next_file()):
        A.define_keypoints()
        A.augment(augmentation_sequence)
        A.plotter(marks=True, raw=False)
        A.save_maps()

    Output
    ------

    """
    def __init__(self, folder_in, folder_out, perform_augment=True, swap_axes=False):
        """
        Parameters
        ----------
        folder_in: os.path
                   path where to be augmented files are stored

        folder_out: os.path
                    Path to folder where augmented data is saved

        perform_augment: bool
                         If False no augmentation will be performed

        swap_axes: bool
                   If True, all augmentations are performed twice,
                   once with swaped axes and once with the original axis
        """

        # Define whether or not augmentation is performed
        self.perform_augment = perform_augment
        self.swap_axes = swap_axes

        # initialize data variables
        self.L = None
        self.R = None
        self.B = None
        self.raw_data = None
        self.Lx = None
        self.Ly = None
        self.Rx = None
        self.Ry = None
        self.Bx = None
        self.By = None

        # initialize lists for augmented maps
        self.A_raw_data = []
        self.A_Lx = []
        self.A_Ly = []
        self.A_Rx = []
        self.A_Ry = []
        self.A_Bx = []
        self.A_By = []
        self.A_border_x = []
        self.A_border_y = []

        # initialize Keypoint lists
        self.Keypoints_L = []
        self.Keypoints_R = []
        self.Keypoints_B = []

        # plot instances
        self.ax = None
        self.colors = ['red', 'blue', 'brown']

        # path where files are stored
        self.path_out = folder_out
        # get all file paths (there are not subfolders in marked)
        self.filepaths = [os.path.join(folder_in, x) for x in os.listdir(folder_in) if
                          x.endswith('_data.pkl')]
        self.files_L = [os.path.join(folder_in, x) for x in os.listdir(folder_in) if
                        x.endswith('_L.pkl')]
        self.files_R = [os.path.join(folder_in, x) for x in os.listdir(folder_in) if
                        x.endswith('_R.pkl')]
        self.files_B = [os.path.join(folder_in, x) for x in os.listdir(folder_in) if
                        x.endswith('_B.pkl')]

    def define_keypoints(self):
        """
        self.Keypoints = [[keypoint, keypoint], ...]
        """
        self.Keypoints_L = []
        self.Keypoints_R = []
        self.Keypoints_B = []

        # make sure that keypoints are at the right position within the embedding
        dim = np.shape(self.raw_data)[0]

        for index, row in self.L.iterrows():
            keypoints = []
            for k in range(len(row['Lx'])):
                keypoints.append(ia.Keypoint(x=row['Lx'][k]+dim, y=row['Ly'][k]+dim))
            self.Keypoints_L.append(keypoints)

        for index, row in self.R.iterrows():
            keypoints = []
            for k in range(len(row['Rx'])):
                keypoints.append(ia.Keypoint(x=row['Rx'][k]+dim, y=row['Ry'][k]+dim))
            self.Keypoints_R.append(keypoints)

        if not self.B.empty:
            for index, row in self.B.iterrows():
                keypoints = []
                for k in range(len(row['Bx'])):
                    keypoints.append(ia.Keypoint(x=row['Bx'][k]+dim, y=row['By'][k]+dim))
                self.Keypoints_B.append(keypoints)

    def augment(self, seqs):
        """
        Performs augmentation sequence given by seqs.

        Parameters
        ----------

        seqs: list
              list of imgaug augmenters
        """
        # reset data lists
        self.A_raw_data = []
        self.A_Lx = []
        self.A_Ly = []
        self.A_Rx = []
        self.A_Ry = []
        self.A_Bx = []
        self.A_By = []
        self.A_border_x = []
        self.A_border_y = []
        # append original non augmented data
        self.A_raw_data.append(self.raw_data)
        self.A_Lx.append(self.L['Lx'])
        self.A_Ly.append(self.L['Ly'])
        self.A_Rx.append(self.R['Rx'])
        self.A_Ry.append(self.R['Ry'])
        self.A_Bx.append(self.B['Bx'])
        self.A_By.append(self.B['By'])
        self.A_border_x.append([0, 0, len(self.raw_data[:, 0]), len(self.raw_data[:, 0])])
        self.A_border_y.append([0, len(self.raw_data[0, :]), len(self.raw_data[0, :]), 0])

        if self.perform_augment:
            # embed data in a larger numpy array filled with zeros
            (x_dim, y_dim) = np.shape(self.raw_data)

            embd_raw = np.zeros((x_dim*3, y_dim*3))
            embd_raw[x_dim:2*x_dim, y_dim:2*y_dim] = np.array(self.raw_data)
            # iterate through augmentation sequences
            seq_len = len(seqs)

            # prevent adding to the same list over and over again
            end_seqs = seqs.copy()
            if self.swap_axes:
                swap = iaa.Sequential(
                    [
                        iaa.Affine(rotate=(90, 90)),
                        iaa.Flipud(1),
                    ])
                # only swap and do nothing else
                end_seqs.append(swap)

                # add the swaped augmentations as well
                for k in range(seq_len):
                    swap_seq = iaa.Sequential(
                        [
                            swap,
                            seqs[k]
                        ])
                    end_seqs.append(swap_seq)

            for k, seq in enumerate(end_seqs):
                seq_det = seq.to_deterministic()
                A_Lx_i = []
                A_Ly_i = []
                A_Rx_i = []
                A_Ry_i = []
                A_Bx_i = []
                A_By_i = []

                self.A_raw_data.append(seq_det.augment_image(embd_raw))

                # augment Left transition
                for keypoints in self.Keypoints_L:
                    aug_keypoints = seq_det.augment_keypoints(ia.KeypointsOnImage(keypoints, shape=np.shape(embd_raw)))
                    coords_array = aug_keypoints.to_xy_array()
                    A_Lx_i.append(coords_array[:, 0])
                    A_Ly_i.append(coords_array[:, 1])

                # augment right transition
                for keypoints in self.Keypoints_R:
                    aug_keypoints = seq_det.augment_keypoints(ia.KeypointsOnImage(keypoints, shape=np.shape(embd_raw)))
                    coords_array = aug_keypoints.to_xy_array()
                    A_Rx_i.append(coords_array[:, 0])
                    A_Ry_i.append(coords_array[:, 1])

                    # augment Left transition
                for keypoints in self.Keypoints_B:
                    aug_keypoints = seq_det.augment_keypoints(ia.KeypointsOnImage(keypoints, shape=np.shape(embd_raw)))
                    coords_array = aug_keypoints.to_xy_array()
                    A_Bx_i.append(coords_array[:, 0])
                    A_By_i.append(coords_array[:, 1])
                # transform borders as well
                border_key = []
                cords = [(x_dim, y_dim),
                         (x_dim, 2*y_dim-1),
                         (2*x_dim-1, 2*y_dim-1),
                         (2*x_dim-1, y_dim)]

                for (x, y) in cords:
                    border_key.append(ia.Keypoint(x=x, y=y))
                corner_aug = seq_det.augment_keypoints(
                    ia.KeypointsOnImage(border_key, shape=np.shape(embd_raw))
                ).to_xy_array()

                # Write the augmented points into list

                self.A_Bx.append(A_Bx_i)
                self.A_By.append(A_By_i)

                # make sure that for swapped images the left & right transitions are changed as well
                if k < seq_len:
                    self.A_border_x.append(corner_aug[:, 0])
                    self.A_border_y.append(corner_aug[:, 1])
                    self.A_Lx.append(A_Lx_i)
                    self.A_Ly.append(A_Ly_i)
                    self.A_Rx.append(A_Rx_i)
                    self.A_Ry.append(A_Ry_i)

                # change left & right transitions
                else:
                    # change order of points (somehow the obvious A_Rx_i[::-1]
                    # does not work
                    A_Rx_i_rev = []
                    for el in A_Rx_i:
                        A_Rx_i_rev.append(el[::-1])
                    A_Ry_i_rev = []
                    for el in A_Ry_i:
                        A_Ry_i_rev.append(el[::-1])
                    A_Ly_i_rev = []
                    for el in A_Ly_i:
                        A_Ly_i_rev.append(el[::-1])
                    A_Lx_i_rev = []
                    for el in A_Lx_i:
                        A_Lx_i_rev.append(el[::-1])

                    rolled_corner = np.roll(corner_aug, 1, axis=0)
                    self.A_border_x.append(rolled_corner[::-1, 0])
                    self.A_border_y.append(rolled_corner[::-1, 1])
                    self.A_Lx.append(A_Rx_i_rev)
                    self.A_Ly.append(A_Ry_i_rev)
                    self.A_Rx.append(A_Lx_i_rev)
                    self.A_Ry.append(A_Ly_i_rev)

    def plotter(self, marks=True, raw=False):
        """
        Plots the already defined lines

        Parameters
        ----------
        marks: bool (optional)
               Default: True
               If True transition lines are plotted

        raw: bool (optional)
             Default: False
             If True, raw data is plotted.
             If False, the derivative of the data is plotted.
        """
        norm = colors.Normalize(vmin=-3, vmax=3)
        for l in range(len(self.A_raw_data)):
            data = x_y_derivator(self.A_raw_data[l])
            fig, self.ax = plt.subplots()
            self.ax.set_ylim(np.shape(self.A_raw_data[l])[1] - 2, 0)
            self.ax.set_xlim(0, np.shape(self.A_raw_data[l])[0] - 2)

            filename = os.path.split(self.filepaths[0])[-1]
            filename = filename.split('.')[0]
            filename = filename.split('_')[0]
            filename = '{}_{}'.format(filename, l)
            if raw:
                dim = len(np.array(self.A_raw_data[l])[:, 0])
                x_axis = np.arange(0, dim)
                y_axis = np.arange(0, dim)
                plt.pcolormesh(x_axis, y_axis, self.A_raw_data[l], norm=norm)
            else:
                x_axis = np.arange(0, len(data[:, 0]))
                y_axis = np.arange(0, len(data[0, :]))
                plt.pcolormesh(x_axis, y_axis, data, norm=norm)

            if marks:
                # plot L transitions
                for k in range(len(self.A_Lx[l])):
                    self.ax.plot(self.A_Lx[l][k], self.A_Ly[l][k], 'x', color=self.colors[0])
                    self.ax.plot(self.A_Lx[l][k], self.A_Ly[l][k], color=self.colors[0])

                # plot R transitions
                for k in range(len(self.A_Rx[l])):
                    self.ax.plot(self.A_Rx[l][k], self.A_Ry[l][k], 'x', color=self.colors[1])
                    self.ax.plot(self.A_Rx[l][k], self.A_Ry[l][k], color=self.colors[1])

                # plot boundaries
                for k in range(len(self.A_Bx[l])):
                    self.ax.plot(self.A_Bx[l][k], self.A_By[l][k], 'x', color=self.colors[2])
                    self.ax.plot(self.A_Bx[l][k], self.A_By[l][k], color=self.colors[2])

            if self.A_border_x:  # if list A_border_x not empty
                self.ax.plot(self.A_border_x[l], self.A_border_y[l], 'x', color=self.colors[2])
                self.ax.plot(self.A_border_x[l], self.A_border_y[l], color=self.colors[2])
                self.ax.plot([self.A_border_x[l][0], self.A_border_x[l][-1]],
                             [self.A_border_y[l][0], self.A_border_y[l][-1]],
                             color=self.colors[2])

            plt.title(filename)
            plt.show()

    def next_file(self):
        """
        Deines what files to read in next by the method read_files_()
        """

        if len(self.filepaths) == 1:
            return False
        else:
            if self.raw_data is not None:
                self.filepaths = self.filepaths[1:]
                self.files_L = self.files_L[1:]
                self.files_R = self.files_R[1:]
                self.files_B = self.files_B[1:]

            self.read_files_()

            return True

    def save_maps(self):
        """
        Saves augmented maps to self.path_out
        """
        for l in range(len(self.A_raw_data)):
            dfL = pd.DataFrame()
            dfR = pd.DataFrame()
            dfB = pd.DataFrame()
            dfBorder = pd.DataFrame()
            df_raw_data = pd.DataFrame(self.A_raw_data[l])
            dfL['Lx'] = self.A_Lx[l]
            dfL['Ly'] = self.A_Ly[l]
            dfR['Rx'] = self.A_Rx[l]
            dfR['Ry'] = self.A_Ry[l]
            dfB['Bx'] = self.A_Bx[l]
            dfB['By'] = self.A_By[l]
            dfBorder['x'] = self.A_border_x[l]
            dfBorder['y'] = self.A_border_y[l]
            self.Lx = []
            self.Ly = []
            self.Rx = []
            self.Ry = []
            self.Bx = []
            self.By = []
            filename = os.path.split(self.filepaths[0])[-1]
            filename = filename.split('.')[0]
            map = filename.split('_')[0]
            filename = '{}_{}'.format(map, l)

            path = os.path.join(self.path_out, map)
            if not os.path.exists(path):
                os.makedirs(path)

            dfL.to_pickle(path + '/{}_L.pkl'.format(filename))
            dfR.to_pickle(path + '/{}_R.pkl'.format(filename))
            dfB.to_pickle(path + '/{}_B.pkl'.format(filename))
            dfBorder.to_pickle(path + '/{}_border.pkl'.format(filename))
            df_raw_data.to_pickle(path + '/{}_data.pkl'.format(filename))

    def read_files_(self):
        """
        Reads all files from self.filepaths[0]
        """
        self.L = pd.read_pickle(self.files_L[0])
        self.R = pd.read_pickle(self.files_R[0])
        if not self.files_B == []:
            self.B = pd.read_pickle(self.files_B[0])
        else:
            self.B = pd.DataFrame()
        self.raw_data = pd.read_pickle(self.filepaths[0])
        # rescale data
        self.raw_data = np.array(scaler(self.raw_data))


if __name__ == '__main__':
    # define augmentation transformations
    swap = iaa.Sequential(
        [
            iaa.Affine(rotate=(90, 90)),
            iaa.Flipud(1),
        ])

    rot = iaa.Sequential(
        [
            iaa.Affine(rotate=(-8, 8)),
        ])

    inc_capacities = iaa.Sequential(
        [
            iaa.Affine(rotate=(45, 45)),
            iaa.Affine(scale={"x": (0.6, 1.7), "y": (0.6, 1.7)}),  # before (0.7, 1.5)
            iaa.Affine(rotate=(-45, -45))
        ])

    scale = iaa.Sequential(
        [
            iaa.Affine(scale={"x": (0.6, 1.7), "y": (0.6, 1.7)}),  # before (0.65, 1.4)
        ]
    )

    # put the above transformations together and apply between 1 to all of them
    seq = iaa.SomeOf((1, None),
                     [
                         rot,
                         inc_capacities,
                         scale
                     ]
                     )

    # for swap_axes=True twice as many augmentation sequences will be performed
    # i.e. the same with swaped axes as well
    augmentation_sequence = [seq]*8

    input_folder = '../../data/coarse/train/not_augmented/'
    output_folder = '../../data/coarse/train/augmented/'

    A = Augmenter(input_folder, output_folder, perform_augment=True, swap_axes=True)
    then = time.time()
    while A.next_file():
        A.define_keypoints()
        A.augment(augmentation_sequence)
        # A.plotter(marks=True, raw=False)
        A.save_maps()
    print('The augmentation took: {}s'.format(time.time() - then))
