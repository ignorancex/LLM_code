"""
Copyright (C) 2023 dwh GmbH - All Rights Reserved
You may use, distribute and modify this code under the 
terms of the MIT license.

You should have received a copy of the MIT license with
this file. If not, please write to: 
martin.bicher@dwh.at or visit 
https://github.com/dwhGmbH/covid19_model_family/blob/main/LICENSE.txt
"""

import csv
import os
import sys

from calibrate import Calibrate
from case_numbers import CaseNumbers
from config import Config
from forecast import Forecast
from hospital_numbers import HospitalNumbers
import datetime as dt


def check_time_series(config: Config) -> None:
    """
    Simply checks whether the setup parameters in the config are compatible with the input data
    Raises errors in case of conflicts
    :param config: config instance
    :return:
    """
    caseNumbers = CaseNumbers(config)
    hospNumbers = HospitalNumbers(config)
    msg = lambda i, daystr, day, notin: 'column ' + str(i) + ': ' + daystr + ' ' + day.strftime(
        '%Y-%m-%d') + ' not in ' + notin
    for i in range(config.get_column_number()):
        if not config.get_start_transient_day() in caseNumbers.get_times(i):
            raise ValueError(msg(i, 'start day of transient phase', config.get_start_transient_day(), 'case numbers'))
        if not config.get_start_calibration_day() in caseNumbers.get_times(i):
            raise ValueError(
                msg(i, 'start day of calibration phase', config.get_start_calibration_day(), 'case numbers'))
        if not config.get_forecast_day() in caseNumbers.get_times(i):
            raise ValueError(msg(i, 'start day of forecast', config.get_forecast_day(), 'case numbers'))
        if not config.get_forecast_day() + dt.timedelta(1) in caseNumbers.get_times(i):
            raise ValueError(
                msg(i, 'first day of forecast', config.get_forecast_day() + dt.timedelta(1), 'case numbers'))
        if not config.get_end_forecast_day() - dt.timedelta(1) in caseNumbers.get_times(i):
            print('Warning: ' + msg(i, 'end day of forecast', config.get_end_forecast_day() - dt.timedelta(1),
                                    'case numbers'))
        if not config.get_start_transient_day() in hospNumbers.get_times(i):
            raise ValueError(
                msg(i, 'start day of transient phase', config.get_start_transient_day(), 'hospital numbers'))
        if not config.get_start_calibration_day() in hospNumbers.get_times(i):
            raise ValueError(
                msg(i, 'start day of calibration phase', config.get_start_calibration_day(), 'hospital numbers'))
        if not config.get_forecast_day() in hospNumbers.get_times(i):
            raise ValueError(msg(i, 'start day of forecast', config.get_forecast_day(), 'hospital numbers'))


def export_result(config: Config, Result: dict) -> None:
    """
    exports the simulation result to a csv file
    :param config: config instance
    :param Result: result of the simulation as dict object
    :return:
    """
    X = list()
    hdr = ['time']
    hdr += [str(config.get_columns_hospitalized()[i]) for i in range(config.get_column_number())]
    X.append(hdr)
    for i in range(len(Result['time'])):
        line = [Result['time'][i].strftime('%Y-%m-%d')]
        line += [str(Result[k][i]) for k in range(config.get_column_number())]
        X.append(line)
    with open(config.get_result_folder() + '/result.csv', 'w', newline='') as f:
        w = csv.writer(f, delimiter=';')
        w.writerows(X)


if __name__ == '__main__':
    """
    Routine to perform a hospital simulation
    Use the path to the config file or a path to a folder with config files as input
    """
    fl = sys.argv[1]
    if os.path.isdir(fl):
        files = [os.path.join(fl,x) for x in os.listdir(fl) if x.startswith('config') and x.endswith('.json')]
    elif fl.endswith('.json'):
        files = [fl]
    else:
        raise RuntimeError('Cannot run simulation. Specified config file or folder is not valid')
    for file in files:
        print('run: ' + file)
        config = Config(file)
        scenario = config.get_scenario()
        # check whether all time series are compatible:
        check_time_series(config)
        # calibration
        Calibrate(config).calibrate()  # writes solution of calibration into config
        # forecast
        Result = Forecast(config).forecast()
        # write result into file
        export_result(config, Result)