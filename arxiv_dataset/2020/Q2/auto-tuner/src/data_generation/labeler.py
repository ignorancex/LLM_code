"""
Author: Renato Durrer
Created: 05.04.2019

Class used to create and label frames.
Files taken from folder data/augmented
Output folder data/labelled
"""

import numpy as np
import pandas as pd
import os
from random import randint
from shapely.geometry import LineString, Point
from shapely.geometry.polygon import Polygon
import matplotlib.pyplot as plt
from matplotlib import colors
from src.utils.funcs import x_y_derivator, subtract_median, preprocessor
import time


class Labeler:
    """
    Class to create, label & process subframes for the task
    of recognizing charge transitions.
    All maps located in subfolders of folder_path are labeled,
    processed and saved in output_folder as a pickle file.


    Workflow Example
    ----------------
    label = Labeler(path_to_folder, path_out, p, f)
    while(label.next_file()):
        label.create_shapes()
        label.create_frames(10)
        label.plotter()
        label.save_frames()

    Output
    ------
    The output files saved in output_folder are structured as follows.

    path: output_folder/m/m_n.pkl
    where: m: map number (~name)
    n: augmentation number (0 = identity)

    File structure:
    The file is a pickled pandas.DataFrame
    Columns:
    frame : np.array(), np.shape(f+2p, f+2p)
            processed subframes. "data" with which the models will be learned
    label : list
            list of tuples indicating a charge transition for
            each dot for the three corners of the frame.
            [(1, 0), (1,1), (0,1)]
             right-top_right-left
    is_feas : int
              indicates wheter or not frame is in feasible reagion.
              1: feasible region
              0: not feasible region
    """
    def __init__(self, folder_path, output_folder, p, f):
        """
        Parameters
        ----------
        folder_path : os.path object
                      folder where files are read
        output_folder : os.path object
                        folder where files are saved
        p : int
            padding
        f : int
            frame size
        """
        self.folderpaths = [os.path.normpath((os.path.join(folder_path, x)))
                            for x in os.listdir(folder_path) if not x.endswith('.gitkeep')]
        self.output_folder = output_folder

        # list of data files for last folderpath
        self.filepaths = None
        self.files_L = None
        self.files_R = None
        self.files_B = None
        self.files_border = None
        self.filename = None

        # contains data to plot
        self.raw_data = None
        self.data = None
        # DataFrame containing all coordinates for Left/Right dot charge transitions
        # created by get_files()
        self.L = None
        self.R = None
        self.B = None
        self.border = None
        self.boundaries = None
        # list of shapely objects for Left/Right dot charge transitions
        # created by create_shapes()
        self.L_transitions = None
        self.R_transitions = None
        self.n_regions = None

        # define frame size
        self.p = p
        self.f = f
        self.frames = None

        # define plot instance
        self.ax = None

    def create_shapes(self):
        """
        Given self.L & self.R, creates shapely objects for dot transition.
        Those objects are saved in self.L_transitions & self.R_transitions.
        Additionally if a non feasible region is defined, a Polygon is created
        defining this non feasible region. The Polygon is stored in self.region
        """
        self.L_transitions = []
        self.R_transitions = []
        self.n_regions = []
        for s in range(len(self.L['Lx'])):
            tuples = [(self.L['Lx'][s][k], self.L['Ly'][s][k]) for k in range(len(self.L['Lx'][s]))]
            self.L_transitions.append(LineString(tuples))

        for s in range(len(self.R['Rx'])):
            tuples = [(self.R['Rx'][s][k], self.R['Ry'][s][k]) for k in range(len(self.R['Rx'][s]))]
            self.R_transitions.append(LineString(tuples))

        for s in range(len(self.B['Bx'])):
            tuples = [(self.B['Bx'][s][k], self.B['By'][s][k]) for k in range(len(self.B['Bx'][s]))]
            self.n_regions.append(Polygon(tuples))

        # create polygon within frames can be drawn
        self.boundaries = Polygon([(self.border['x'][0], self.border['y'][0]),
                                   (self.border['x'][1], self.border['y'][1]),
                                   (self.border['x'][2], self.border['y'][2]),
                                   (self.border['x'][3], self.border['y'][3])])

    def create_frames(self, rand):
        """
        Creates self.frames : a list of tuples (frames, labels)
        Example:
        [(x_0, label_0, is_feas_0, shapes_0), (x_1, label_1, is_feas_1, shapes_1), ...]

        with x_i: np.array() <- shape(n, n)
        label_i: list of tuples indicating a charge transition for
                 each dot for the three corners of the frame.
                 [(1, 0), (1,1), (0,1)]
                  right-top_right-left
        is_feas_i: int
                   indicates wheter or not frame is in feasible reagion.
                   1: feasible region
                   0: not feasible region
        shapes_i: shapely shapes of the frame boundaries
                  for plotting reasons

        It is
        n = f + 2p

        Parameters
        ----------
        rand : int
            randomness, the acutal frame will deviate from
            the grid by a random number drawn from [-rand/2, rand/2]
        """

        w, h = np.shape(self.data)
        self.frames = []
        if w > 300:
            x_v = np.arange(int(w/6), int(5*w/6), self.f)
            y_v = np.arange(int(h/6), int(5*h/6), self.f)
        else:
            x_v = np.arange(self.p, w-self.p, self.f)
            y_v = np.arange(self.p, h-self.p, self.f)
        x_p, y_p = np.meshgrid(x_v, y_v)

        for (x_g, y_g) in np.nditer((x_p, y_p)):
            x = int(x_g + randint(-rand/2, rand/2))
            y = int(y_g + randint(-rand/2, rand/2))

            # generate corners of the frames including padding
            tl = Point(x-self.p, y-self.p)
            tr = Point(x+self.p+self.f+1, y-self.p)
            bl = Point(x-self.p, y+self.p+self.f+1)
            br = Point(x+self.p+self.f+1, y+self.p+self.f+1)

            # check whether frame lies within data
            if all(self.boundaries.contains(l) for l in [tl, tr, bl, br]):
                # create the data frame
                raw_frame = self.raw_data[y-self.p:y+self.p+self.f+1, x-self.p:x+self.p+self.f+1]
                # reorder the points. The ordering is wrong due to how the measurement is taken
                raw_frame = raw_frame[::-1, :]

                if np.all(raw_frame == 0):
                    print('x: {}, y: {}\n'
                          'shape: {}'.format(x, y, np.shape(raw_frame)))
                    print(self.filepaths[0])

                frame = preprocessor(raw_frame)
                # create shapely lines
                top = LineString([(x, y), (x+self.f, y)])
                left = LineString([(x, y), (x, y+self.f)])
                right = LineString([(x+self.f, y), (x+self.f, y+self.f)])
                bottom = LineString([(x, y+self.f), (x+self.f, y+self.f)])
                label = []
                # label the frames
                label.append((any(l.crosses(bottom) for l in self.L_transitions),
                              any(l.crosses(bottom) for l in self.R_transitions)))
                # following line needs review
                label.append((any(l.crosses(bottom) or l.crosses(right) for l in self.L_transitions),
                              any(l.crosses(bottom) or l.crosses(right) for l in self.R_transitions)))

                label.append((any(l.crosses(left) for l in self.L_transitions),
                              any(l.crosses(left) for l in self.R_transitions)))
                if self.n_regions is None:
                    is_feas = True
                else:
                    is_feas = all(l.disjoint(k) for l in [top, left, right, bottom] for k in self.n_regions)

                self.frames.append([frame, label, is_feas, (top, left, right, bottom)])

    def plotter(self, marks=True):
        """
        Plots shapely lines, processed data plus the drawn frames including labels.

        Parameters
        ----------

        marks: bool
               If true, the charge indication lines are plotted as well.
        """
        fig, self.ax = plt.subplots()
        # plot marks / shapely lines
        x_axis = np.arange(0, np.shape(self.data)[0])
        y_axis = np.arange(0, np.shape(self.data)[1])

        self.ax.set_ylim(np.shape(self.data)[1] - 1, 0)
        self.ax.set_xlim(0, np.shape(self.data)[0] - 1)
        norm = colors.Normalize(vmin=-3, vmax=3)
        self.ax.pcolormesh(x_axis, y_axis, self.data, norm=norm)
        if marks:
            for k in range(len(self.L_transitions)):
                x, y = self.L_transitions[k].xy
                self.ax.plot(x, y, color='red')
            for k in range(len(self.R_transitions)):
                x, y = self.R_transitions[k].xy
                self.ax.plot(x, y, color='blue')
            for k in range(len(self.n_regions)):
                if not self.n_regions[0].area == 0:
                    x, y = self.n_regions[k].boundary.xy
                    self.ax.plot(x, y, color='brown')

        if not self.border.empty:
            x, y = self.boundaries.boundary.xy
            self.ax.plot(x, y, color='black')

        # plot frames
        x_0 = None
        y_0 = None
        for frame in self.frames:
            for k, line in enumerate(frame[3]):
                x, y = line.xy
                self.ax.plot(x, y, color='black')
                if k == 0:
                    x_0 = x[0]
                    y_0 = y[0]

            plt.annotate('({},{})'.format(int(frame[1][0][0]), int(frame[1][0][1])), (x_0+self.f, y_0+self.f),
                         fontsize=8)
            plt.annotate('({},{})'.format(int(frame[1][1][0]), int(frame[1][1][1])), (x_0+self.f, y_0), fontsize=8)
            plt.annotate('({},{})'.format(int(frame[1][2][0]), int(frame[1][2][1])), (x_0, y_0), fontsize=8)
            plt.annotate(int(frame[2]), (x_0, y_0+self.f), fontsize=8)
        filename = os.path.split(self.filepaths[0])[-1]
        filename = filename.split('.')[0]

        plt.title(filename)
        plt.draw()

    def save_frames(self):
        """
        Saves the drawn frames from one map as a pandas.DataFrame object which is pickled.
        Thereby, the data from every frame is rescaled.
        columns:
        frames, label, is_feas
        """
        np_frames = np.array(self.frames)
        df_frames = pd.DataFrame()
        df_frames['frames'] = np_frames[:, 0]
        df_frames['label'] = np_frames[:, 1]
        df_frames['is_feas'] = np_frames[:, 2]

        filename = os.path.split(self.filepaths[0])[-1]
        filename = filename.split('.')[0]
        filename = filename.split('_')
        folder_path = os.path.join(self.output_folder, filename[0])
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        file_path = os.path.join(folder_path, '{}_{}.pkl'.format(filename[0], filename[1]))
        df_frames.to_pickle(file_path)

    def next_file(self):
        """
        Reads the next map from input_folder.
        """
        if self.filepaths is not None:
            self.filepaths = self.filepaths[1:]
            self.files_B = self.files_B[1:]
            self.files_L = self.files_L[1:]
            self.files_R = self.files_R[1:]
            self.files_border = self.files_border[1:]

        if self.filepaths == [] or self.filepaths is None:
            if self.filepaths is not None:
                self.folderpaths = self.folderpaths[1:]

            if not self.folderpaths:  # check whether list folderpaths is empty
                print('All files processed')
                return False

            else:
                self.filepaths = [(os.path.join(self.folderpaths[0], x)) for x in os.listdir(self.folderpaths[0]) if
                                  x.endswith('_data.pkl')]
                self.files_L = [os.path.join(self.folderpaths[0], x) for x in os.listdir(self.folderpaths[0]) if
                                x.endswith('_L.pkl')]
                self.files_R = [os.path.join(self.folderpaths[0], x) for x in os.listdir(self.folderpaths[0]) if
                                x.endswith('_R.pkl')]
                self.files_B = [os.path.join(self.folderpaths[0], x) for x in os.listdir(self.folderpaths[0]) if
                                x.endswith('_B.pkl')]
                self.files_border = [os.path.join(self.folderpaths[0], x) for x in os.listdir(self.folderpaths[0]) if
                                     x.endswith('_border.pkl')]

        # Read files
        self.L = pd.read_pickle(self.files_L[0])
        self.R = pd.read_pickle(self.files_R[0])
        self.B = pd.read_pickle(self.files_B[0])
        self.border = pd.read_pickle(self.files_border[0])

        # read raw data
        self.raw_data = pd.read_pickle(self.filepaths[0])

        # process data for plotting purpose
        self.raw_data = np.array(self.raw_data)
        self.data = subtract_median(x_y_derivator(self.raw_data))
        return True


if __name__ == "__main__":
    path_to_folder = '../../data/fine/train/augmented/'
    path_out = '../../data/fine/train/labeled_scaled/'

    then = time.time()
    label = Labeler(path_to_folder, path_out, 8, 12)
    while label.next_file():
        label.create_shapes()
        label.create_frames(10)
        # label.plotter(marks=True)
        label.save_frames()
    print('The labeling process took {}s'.format(time.time()-then))
