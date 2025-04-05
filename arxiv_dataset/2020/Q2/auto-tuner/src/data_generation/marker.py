"""
Author: Renato Durrer
Created: 05.04.2019

File used to define charge transition in the full resolution maps
"""
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import numpy as np
import pandas as pd
import Labber
from src.utils.funcs import x_y_derivator, data_creator


class PictureIterator:
    """
    Class with which charge transitions can be indicated on measured charge stability diagrams.
    The charge transition lines are saved as pickle files.
    """
    def __init__(self, folderpath, path_out, ax, normalize=False):
        """
        Parameters
        ----------
        folderpath : os.path
                     path to folder with files that should get marked

        path_out: os.path
                  path where pickle files are saved

        ax : pyplot.ax object
             Axe with which plots are generated

        normalize: bool
            whether or not colorscale is renormalized
        """

        self.folderpath = folderpath
        # The variable filepath will contain the paths to all the .hdf5 files (that do not end with _cal.hdef5) in the
        # folder found under folderpath
        self.filepaths = [os.path.normpath(os.path.join(self.folderpath, x))
                          for x in os.listdir(self.folderpath) if (not x.endswith('_cal.hdf5') and x.endswith('.hdf5'))]
        self.path_out = path_out

        # kind == 1 <-> 'red' corresponds to the dot associated with the Plunger Gate on the x-axis
        # kind == 0 <-> 'blue' corresponds to the dot /âssociated Plunger Gate on the y-axis
        # our convention is, that the y-plunger gate corresponds to the right
        self.colors = ['red', 'blue', 'brown']
        self.kind = 0

        # initialize storage for coordinates
        self.Lx = []
        self.Ly = []
        self.Rx = []
        self.Ry = []
        self.Bx = []
        self.By = []

        # Initialize x and y as None
        self.x = None
        self.y = None
        self.data = None
        self.raw_data = None

        # initialize frame information
        self.image_height = None
        self.image_width = None

        # plot variables
        self.ax = ax
        self.points = None
        self.lines = None
        self.log = False
        self.normalize = normalize
        self.norm = mpl.colors.Normalize()

        # zoom information
        self.xdata = None
        self.ydata = None
        self.scale_factor = None
        self.cur_xrange = None
        self.cur_yrange = None

        # text location in axes coords
        # Connect the Button press even with the onclick function
        plt.connect('button_press_event', self.onclick)
        plt.connect('key_press_event', self.on_key)
        plt.connect('scroll_event', self.scroll)

    def outputpickle(self):
        """
        saves lines in a pickle file
        """
        if not self.x == []:
            if self.kind == 0:
                self.Lx.append(self.x)
                self.Ly.append(self.y)
            if self.kind == 1:
                self.Rx.append(self.x)
                self.Ry.append(self.y)
            if self.kind == 2:
                self.Bx.append(self.x)
                self.By.append(self.y)

        dfL = pd.DataFrame()
        dfR = pd.DataFrame()
        dfB = pd.DataFrame()
        df_raw_data = pd.DataFrame(self.raw_data)
        dfL['Lx'] = self.Lx
        dfL['Ly'] = self.Ly
        dfR['Rx'] = self.Rx
        dfR['Ry'] = self.Ry
        dfB['Bx'] = self.Bx
        dfB['By'] = self.By
        self.Lx = []
        self.Ly = []
        self.Rx = []
        self.Ry = []
        self.Bx = []
        self.By = []
        filename = os.path.split(self.filepaths[0])[-1]
        filename = filename.split('.')[0]
        path = os.path.join(self.path_out, filename)

        dfL.to_pickle(path + '_L.pkl')
        dfR.to_pickle(path + '_R.pkl')
        dfB.to_pickle(path + '_B.pkl')
        df_raw_data.to_pickle(path + '_data.pkl')

        plt.savefig(os.path.join(self.path_out, filename + '.jpg'))

    def loadPlot(self):
        """
        Function to load new data, processes it and creates the plot
        """
        labber_data = Labber.LogFile(self.filepaths[0])
        raw_data = data_creator(labber_data)
        self.raw_data = raw_data['I QPC'][:, ::-1]  # revert PG1 axis, as it is measured in reverse
        self.data = x_y_derivator(self.raw_data)
        filename = os.path.split(self.filepaths[0])[-1]
        filename = filename.split('.')[0]
        self.image_height = len(raw_data['MPG0'][0, :-1]) - 1
        self.image_width = len(raw_data['LPG0'][:-1, 0]) - 1
        y_axis = np.arange(0, self.image_height+1)
        x_axis = np.arange(0, self.image_width+1)

        if self.normalize:
            upper = np.percentile(self.data, 99.5)
            lower = np.percentile(-1*self.data, 99.5)
            self.norm = mpl.colors.Normalize(vmin=-lower, vmax=upper)
        plt.pcolormesh(x_axis, y_axis, self.data, norm=self.norm)
        self.ax.set_xlim([0, self.image_width])
        self.ax.set_ylim([self.image_height, 0])
        plt.tight_layout()
        plt.title(filename)
        plt.draw()

    def plotter(self):
        """
        Plots the already defined lines
        """
        plt.tight_layout()
        filename = os.path.split(self.filepaths[0])[-1]
        filename = filename.split('.')[0]
        x_axis = np.arange(0, len(self.data[0, :]))
        y_axis = np.arange(0, len(self.data[:, 0]))

        # SymLogNorm does not work -.-
        if self.log:
            plt.pcolormesh(x_axis, y_axis, self.data, norm=mpl.colors.SymLogNorm(linthresh=0.03, linscale=0.03,
                                                                                 vmin=-1.0, vmax=1.0))
        else:
            plt.pcolormesh(x_axis, y_axis, self.data, norm=self.norm)

        # plot L transitions
        for k in range(len(self.Lx)):
            self.ax.plot(self.Lx[k], self.Ly[k], 'x', color=self.colors[0])
            self.ax.plot(self.Lx[k], self.Ly[k], color=self.colors[0])

        # plot R transitions
        for k in range(len(self.Rx)):
            self.ax.plot(self.Rx[k], self.Ry[k], 'x', color=self.colors[1])
            self.ax.plot(self.Rx[k], self.Ry[k], color=self.colors[1])

        # plot boundaries
        for k in range(len(self.Bx)):
            self.ax.plot(self.Bx[k], self.By[k], 'x', color=self.colors[2])
            self.ax.plot(self.Bx[k], self.By[k], color=self.colors[2])

        # plot current points
        self.points, = self.ax.plot(self.x, self.y, 'x', color=self.colors[self.kind])
        self.lines, = self.ax.plot(self.x, self.y, color=self.colors[self.kind])

        # keep zoom
        plt.title(filename)
        plt.draw()

    def on_key(self, event):
        """
        action handler for keys
        Parameters
        ----------
        event: event object
        """
        if event.key == 'c':
            # change the kind of line
            self.kind += 1
            self.kind = self.kind % 3

        if event.key == 'n':
            # creae a new line / linestring instance
            if self.kind == 0:
                self.Lx.append(self.x)
                self.Ly.append(self.y)
            if self.kind == 1:
                self.Rx.append(self.x)
                self.Ry.append(self.y)
            if self.kind == 2:
                self.Bx.append(self.x)
                self.By.append(self.y)
            self.x = []
            self.y = []

        if event.key == 'r':
            # reset the axis to original ones
            self.reset_zoom()

        if event.key == 'd':
            # delete last line of current kind
            if self.kind == 0:
                self.Lx = self.Lx[:-1]
                self.Ly = self.Ly[:-1]
            if self.kind == 1:
                self.Rx = self.Rx[:-1]
                self.Ry = self.Ry[:-1]
            if self.kind == 2:
                self.Bx = self.Bx[:-1]
                self.By = self.By[:-1]
            self.plotter()

        if event.key == 'l':
            # change colorscale logarithmic <-> linear
            self.log = not self.log
            self.plotter()

        # press b. Remove the last point and replot the figure
        if event.key == 'b':
            self.x = self.x[:-1]
            self.y = self.y[:-1]
            self.ax.lines = []
            self.plotter()

    def onclick(self, event):
        """
        Function determening what happens when a click on the mouse is made.

        We check for the following 4 cases.
        1. The click event was not in the area spaned by the figure handle
        2. Left click.
        3. Right click
        4. Middle click

        Parameters
        ----------
        event: handle to the click event

        """
        # If the event happend outside of the figure axis we dont care
        if not event.inaxes:
            return

        # Left click, append the new points and redraw the crosses
        if event.button == 1:
            self.x.append(event.xdata)
            self.y.append(event.ydata)
            self.ax.plot(self.x, self.y, 'x', color=self.colors[self.kind])
            self.ax.plot(self.x, self.y, color=self.colors[self.kind])
            plt.draw()

        # Click on the middle. Remove the last point and replot the figure
        if event.button == 2:
            self.x = self.x[:-1]
            self.y = self.y[:-1]
            self.ax.lines = []
            self.plotter()

        # Click on the right, if x and y are not none (this is only true for)
        # the title picture we store the values in a .pkl and clear the figure
        # and load a new picture
        if event.button == 3:
            if self.x is None:
                self.x = []
                self.y = []
                self.ax.cla()
                self.loadPlot()

            else:
                # self.reset_zoom()
                if self.x:  # check whether list x not empty
                    if self.kind == 0:
                        self.Lx.append(self.x)
                        self.Ly.append(self.y)
                    if self.kind == 1:
                        self.Rx.append(self.x)
                        self.Ry.append(self.y)
                    if self.kind == 2:
                        self.Bx = self.x
                        self.By = self.y
                self.outputpickle()
                self.x = []
                self.y = []
                self.ax.cla()
                self.filepaths.pop(0)

                # if no files are left, exit
                if len(self.filepaths) == 0:
                    plt.close(fig)
                    return True

                else:
                    self.loadPlot()

    def scroll(self, event, base_scale=1.2):
        """
        action handler for scroll events
        -> rescales / zooms
        Parameters
        ----------
        event
        base_scale

        Returns
        -------

        """
        # get the current x and y limits
        cur_xlim = ax.get_xlim()
        cur_ylim = ax.get_ylim()
        self.cur_xrange = (cur_xlim[1] - cur_xlim[0])*.5
        self.cur_yrange = (cur_ylim[1] - cur_ylim[0])*.5
        self.xdata = event.xdata  # get event x location
        self.ydata = event.ydata  # get event y location

        if event.button == 'up':
            # deal with zoom in
            self.scale_factor = 1/base_scale
        elif event.button == 'down':
            # deal with zoom out
            self.scale_factor = base_scale
        else:
            # deal with something that should never happen
            self.scale_factor = 1
            print(event.button)
        # set new limits
        ax.set_xlim([self.xdata - self.cur_xrange*self.scale_factor,
                     self.xdata + self.cur_xrange*self.scale_factor])
        ax.set_ylim([self.ydata - self.cur_yrange*self.scale_factor,
                     self.ydata + self.cur_yrange*self.scale_factor])
        plt.draw()  # force re-draw

    def reset_zoom(self):
        """
        resets zoom to original value
        """
        ax.set_xlim([0,
                     self.image_width])
        ax.set_ylim([self.image_height,
                     0])

        cur_xlim = ax.get_xlim()
        cur_ylim = ax.get_ylim()
        plt.tight_layout()

        self.xdata = self.image_width / 2
        self.ydata = self.image_height / 2
        self.scale_factor = 1
        self.cur_xrange = (cur_xlim[1] - cur_xlim[0])*.5
        self.cur_yrange = (cur_ylim[1] - cur_ylim[0])*.5

        plt.draw()  # force re-draw


if __name__ == "__main__":
    # Initailize a figure with the starting image which explains how to use this
    # program
    fig, ax = plt.subplots()
    ax.text(0.5, 0.5, """Right click to go to the next image\n
            Left click indicate charge transitions \n
            Turning wheel to delete the last mark \n
            "c" for changing type of transition (QD1, QD2, unfeasible area)\n
            "r" for resetting zoom\n
            "n" for new line\n
            "l" for using logarithmic scale\n
            turn turning wheel for zooming
            """,
            horizontalalignment='center',
            verticalalignment='center',
            transform=ax.transAxes,
            bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 10})
    plt.draw()

    # Run the picture iterator class which will do the rest
    filepath = '../../data/fine/train/to_mark/'
    path_out = '../../data/fine/train/marked/'
    pict = PictureIterator(filepath, path_out, ax, normalize=True)
