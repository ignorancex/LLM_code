"""
Author: Renato Durrer
Created: 25.03.2019

File in with helpful functions for data creation and processing are written.
"""
import Labber
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt


def data_creator(logfile):
    """
    Takes the Labber LogFile and creates a new data file for plotting. Supports
    n-dimensional sweeping.

    Parameters
    ----------
    logfile :
        Labber.LogFile() object

    Returns
    -------
    dict
    Example:
    data = {
        'I QPC': np.array() <- shape(n_x, n_y, n_z), # Log
        'I TQD': np.array() <- shape(n_x, n_y, n_z), # Log
        'LPG': np.array() <- shape(n_x, n_y, n_z),   # Step
        'MPG': np.array() <- shape(n_x, n_y, n_z),   # Step
        'LTG': np.array() <- shape(n_x, n_y, n_z)    # Step
    }

    For the case of independent step channels one can get the steps as
    LPG = data['LPG'][:, 0, 0]
    MPG = data['MPG'][0, :, 0]
    LTG = data['LTG'][0, 0, :]

    Attention: the order matters and is given by the logfile!
    """
    # create data dict
    data = {}

    # get shape of the data
    shapes = []
    steps = []
    step_channels = logfile.getStepChannels()
    for channel in step_channels:
        if len(channel['values']) != 1:
            shapes.append(len(channel['values']))
            steps.append(channel['name'])
        else:
            break
    shapes = tuple(shapes)[::-1]

    # find the log channels
    log_channels = logfile.getLogChannels()

    # store the Measurements
    for channel in log_channels:
        if len(log_channels) == 1:
            msm = logfile.getData(name=channel['name'])
            msm = msm.reshape(shapes)
        else:
            msm = logfile.getData(name=channel['name'])
            msm = msm.reshape(shapes)
            msm = np.moveaxis(msm, -1, 0)
            msm = np.moveaxis(msm, -1, 0)
        data[channel['name']] = msm

    # store the step channels
    for channel in steps:
        if len(steps) == 1:
            steps = logfile.getData(name=channel)
            steps = steps.reshape(shapes)
        else:
            steps = logfile.getData(name=channel)
            steps = steps.reshape(shapes)
            steps = np.moveaxis(steps, -2, 0)
            steps = np.moveaxis(steps, -1, 0)
        data[channel] = steps

    return data


def x_y_derivator(data, x=None, y=None):
    """
    Takes 2-dimensional data and returns the superposition of the derivative
    for x and the derivative for y.
    Parameters
    ----------
    data : np.array() <- shape(nx, ny)

    x : np.array() <- shape(nx,), optional

    y : np.array() <- shape(ny,), optional

    Returns
    -------
    np.array() <- shape(nx-1, ny-1)
    """
    if x is None or y is None:
        x = np.arange(0, len(data[:, 0]))
        y = np.arange(0, len(data[0, :]))

    d_measure_x = np.diff(data, axis=0)
    d_measure_y = np.diff(data, axis=1)

    dx = np.diff(x)
    dy = np.diff(y)

    ddx = (d_measure_x.T / dx).T
    ddy = d_measure_y / dy

    # subtract QPC slope
    QPC_x = (data[:, 0] - data[:, -1]) / (x[0] - x[-1])
    QPC_y = (data[0, :] - data[-1, :]) / (y[0] - y[-1])
    x_offset = np.average(QPC_x)
    y_offset = np.average(QPC_y)

    ddx = ddx - x_offset
    ddy = ddy - y_offset

    derv = (ddx[:, :-1] + ddy[:-1, :]) / 2

    return derv


def subtract_median(data):
    """

    Parameters
    ----------
    data

    Returns
    -------

    """
    median = np.median(data)
    return data - median


def scaler(data):
    """

    Parameters
    ----------
    data

    Returns
    -------
    rescaled data
    """
    std = np.std(data)

    return data / std


def remove_outliers(data):
    """
    Remove outliers. An outliers is defined as a point that is more than
    5 standard deviations away from the datas median.

    Parameters
    ----------
    data

    Returns
    -------
    Input data with outliers set to 0.
    """
    upper_boundary = np.quantile(data, 0.992)
    lower_boundary = np.quantile(data, 0.008)
    selection = data[(data > lower_boundary) & (data < upper_boundary)]
    standard_dev = np.std(selection)
    median = np.median(selection)
    data[(median + 4.5 * standard_dev < data) | (data < median - 4.5 * standard_dev)] = median
    return data


def preprocessor(data, rescale=True):
    if rescale:
        return subtract_median(scaler(remove_outliers(x_y_derivator(scaler((data))))))
    else:
        return subtract_median(remove_outliers(x_y_derivator(scaler((data)))))


def collect_data(input_folder, ratio):
    """
    Collects data and puts them in a pandas DataFrame
    Parameters
    ----------
    input_folder : os.path object
                   folder where data is stored

    ratio : float
            describes #learn/#test set.

    Returns
    -------
    pd.DataFrame()
    columns:
    'x': np.array() <- shape(f+2p, f+2p)
    'y': list like <- len(3)

    """
    # TODO implement ratio
    data = pd.DataFrame()

    folderpaths = [os.path.normpath((os.path.join(input_folder, x)))
                   for x in os.listdir(input_folder) if not x.endswith('.gitkeep')]
    # for folder in folderpaths:
    for folder in folderpaths:
        filepaths = [os.path.normpath((os.path.join(folder, x)))
                     for x in os.listdir(folder) if not x.endswith('.gitkeep')]
        for file in filepaths:
            df = pd.read_pickle(file)
            df = df[df['is_feas'] == 1]
            data = data.append(df[['frames', 'label']], ignore_index=True)

    return data.rename(columns={'frames': 'x', 'label': 'y'})
#
#
# def create_encoding(data):
#     """
#
#     Parameters
#     ----------
#     data
#
#     Returns
#     -------
#
#     """
#     labels = data['y']


def create_imgs(folder_in, folder_out):
    """

    Parameters
    ----------
    folder_in : path
        path to folder with .hdf5 files

    folder_out : path
        path to folder where img are saved

    Returns
    -------

    """
    filepaths = [os.path.normpath(os.path.join(folder_in, x))
                 for x in os.listdir(folder_in) if (x.endswith('.hdf5') and not x.endswith('_cal.hdf5'))]

    for file in filepaths:
        labber_data = Labber.LogFile(file)
        raw_data = data_creator(labber_data)
        data = x_y_derivator(raw_data['I QPC'][:, :], raw_data['LPG0'][:, 0], raw_data['MPG0'][0, :])
        filename = os.path.split(file)[-1]
        filename = filename.split('.')[0]
        plt.figure()
        plt.pcolormesh(raw_data['LPG0'][:-1, 0], raw_data['MPG0'][0, :-1], data)
        plt.title(filename)
        plt.savefig(os.path.join(folder_out, filename + '.png'))
