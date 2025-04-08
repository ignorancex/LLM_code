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
from datetime import datetime

from config import Config
from utils import TimeSeries, parse_date


class HospitalNumbers:
    """
    Class for interaction between hospital number file and simulation
    """

    def __init__(self, config: Config) -> None:
        """
        Class for interaction between hospital number files and simulation
        :param config: config instance
        """
        self.beds = {i: TimeSeries(list(), list()) for i in range(config.get_column_number())}
        with open(config.get_filename_hospitalized()) as f:
            r = csv.reader(f, delimiter=';')
            hdr = next(r)
            inds = [hdr.index(x) for x in config.get_columns_hospitalized()]
            for line in r:
                day = parse_date(line[0])
                for i in range(len(inds)):
                    self.beds[i].append_value(day, float(line[inds[i]]))

    def get_beds(self, index: int) -> TimeSeries:
        """
        :param index: index of scenario=column
        :return: TimeSeries object of the hospital numbers for the given index
        """
        return self.beds[index]

    def get_value(self, index: int, time: datetime.date) -> float:
        """
        :param index: index of scenario=column
        :param time: target date
        :return: value at the given date for the given index
        """
        return self.beds[index].get_value(time)

    def get_times(self, index: int) -> list[datetime.date]:
        """
        :param index: index of scenario=column
        :return: list of dates
        """
        return self.beds[index].get_times()

    def get_values(self, index: int) -> list[float]:
        """
        :param index: index of scenario=column
        :return: list of hospital number values
        """
        return self.beds[index].get_values()

    def get_ma_value(self, index: int, time: datetime.date, days: int = 7) -> float:
        """
        Same as "get value" for the backwards moving average
        :param index: index of scenario=column
        :param time: target date
        :param days: number of days to average
        :return: value at the given date for the given index
        """
        return self.beds[index].get_ma_value(time, days)

    def get_ma_values(self, index: int, days: int = 7) -> list[float]:
        """
        Same as "get values" for the backwards moving average
        :param index: index of scenario=column
        :param days: number of days to average
        :return: list of values for the given index
        """
        return self.beds[index].get_ma_values(days)
