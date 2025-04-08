"""
Copyright (C) 2023 dwh GmbH - All Rights Reserved
You may use, distribute and modify this code under the 
terms of the MIT license.

You should have received a copy of the MIT license with
this file. If not, please write to: 
martin.bicher@dwh.at or visit 
https://github.com/dwhGmbH/covid19_model_family/blob/main/LICENSE.txt
"""

from matplotlib import pyplot as plt

from config import Config
from utils import TimeSeries


def plot_calibration(config: Config, data: TimeSeries, simulation: TimeSeries, index: int, title: str) -> None:
    """
    Script to plot the calibration.
    Currently unused
    See "plot_result" for more information
    :param config:
    :param data:
    :param simulation:
    :param index:
    :param title:
    :return:
    """
    N = config.get_column_number()
    if int(N ** 0.5) == N ** 0.5:
        m = N ** 0.5
    else:
        m = int(N ** 0.5) + 1
    n = N / m
    if (n != int(n)):
        n = int(n) + 1
    else:
        n = int(n)
    if index == 0:
        plt.figure(figsize=[16, 9])
    plt.subplot(n, m, index + 1)
    timesttransient = [x for x in data.get_times() if
                       x >= config.get_start_transient_day() and x <= config.get_start_calibration_day()]
    timescalib = [x for x in data.get_times() if
                  x >= config.get_start_calibration_day() and x <= config.get_forecast_day()]
    pl1, = plt.plot(data.get_times(), data.get_values(), 'k')
    pl2, = plt.plot(timesttransient, [simulation.get_value(x) for x in timesttransient], 'r', linestyle='dashed')
    pl3, = plt.plot(timescalib, [simulation.get_value(x) for x in timescalib], 'r')

    span = [config.get_start_transient_day(), config.get_end_forecast_day()]
    plt.xlim(span)
    plt.xticks(rotation=45)
    plt.ylabel('occupancy')
    plt.legend([pl1, pl2, pl3], ['data', 'transient phase', 'calibration phase'])
    plt.title(title)
    if index == config.get_column_number() - 1:
        plt.tight_layout()
        plt.savefig(config.get_result_folder() + '/calibration.png', dpi=300)


def plot_result(config: Config, data: TimeSeries, simulation: TimeSeries, index: int, title: str) -> None:
    """
    Plots the result for the index-th column/scenario of the simulation in a standard line plot.
    Hereby, each scenario will automatically be displayed in one subplot window.
    Calling this function with index = 0 will create the figure
    Calling this function with index = config.get_column_number()-1 will save the plot to a file
    :param config: config instance
    :param data: TimeSeries of the reference hospital occupancy
    :param simulation: TimeSeries of the occupied beds
    :param index: index of the column/scenario to plot
    :param title: identifier of the regarded column/scenario
    :return:
    """
    N = config.get_column_number()
    if int(N ** 0.5) == N ** 0.5:
        m = N ** 0.5
    else:
        m = int(N ** 0.5) + 1
    n = N / m
    if (n != int(n)):
        n = int(n) + 1
    else:
        n = int(n)
    if index == 0:
        plt.figure(figsize=[16, 9])
    plt.subplot(n, m, index + 1)
    timesttransient = [x for x in data.get_times() if
                       x >= config.get_start_transient_day() and x <= config.get_start_calibration_day()]
    timescalib = [x for x in data.get_times() if
                  x >= config.get_start_calibration_day() and x <= config.get_forecast_day()]
    timesforecast = [x for x in simulation.get_times() if
                     x > config.get_forecast_day() and x <= config.get_end_forecast_day()]
    pl1, = plt.plot(data.get_times(), data.get_values(), 'k')
    pl2, = plt.plot(timesttransient, [simulation.get_value(x) for x in timesttransient], 'r', linestyle='dashed')
    pl3, = plt.plot(timescalib, [simulation.get_value(x) for x in timescalib], 'r')
    pl4, = plt.plot(timesforecast, [simulation.get_value(x) for x in timesforecast], 'b')

    span = [config.get_start_transient_day(),
            config.get_end_forecast_day()]
    plt.xlim(span)
    plt.xticks(rotation=45)
    plt.ylabel('occupancy')
    plt.legend([pl1, pl2, pl3, pl4], ['data', 'transient phase', 'calibration phase', 'forecast'])
    plt.title(title)
    if index == config.get_column_number() - 1:
        plt.tight_layout()
        plt.savefig(config.get_result_folder() + '/simResult.png', dpi=300)
