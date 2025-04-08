"""
Copyright (C) 2023 dwh GmbH - All Rights Reserved
You may use, distribute and modify this code under the 
terms of the MIT license.

You should have received a copy of the MIT license with
this file. If not, please write to: 
martin.bicher@dwh.at or visit 
https://github.com/dwhGmbH/covid19_model_family/blob/main/LICENSE.txt
"""

import datetime as dt


class TimeSeries:
    """
    Convenience class to manage timeseries
    """

    def __init__(self, xs, times: list = None, default: float = 0) -> None:
        """
        Creates a scalar time series object
        If times is None and xs is a dictionary, it will create the time series from the dict object.
        Hereby, the keys of the dictionary will be regarded as time basis and the values accordingly as values of the time series
        If times is a list and xs is a list with equal length, the time series will be initialised accordingly
        The default value specifies the value returned if the time series is evaluated at an undefined point in time
        :param xs: list or dict, values must be floats
        :param times: list of equal length as xs or None, values must be hashable
        :param default: float
        """
        if times == None:
            self.data = xs
            self.times = xs.keys()
            self.times.sort()
        else:
            self.times = times
            self.data = {t: x for t, x in zip(times, xs)}
        self.ma_data = dict()
        self.default = default

    def get_value(self, time) -> float:
        """
        evaluate the time series at the requested point in time
        :param time:
        :return:
        """
        try:
            return self.data[time]
        except:
            return self.default

    def get_times(self) -> list:
        """
        returns the time basis of the time series
        :return:
        """
        return list(self.times)

    def get_values(self) -> list[float]:
        """
        returns the values time series. Matches "get_times"
        :return:
        """
        return [self.data[t] for t in self.times]

    def _make_ma(self, days: int = 7) -> None:
        """
        Creates the backwards moving average of the time series.
        Method is called automatically on demand.
        :param days: number of days to average
        :return:
        """
        self.ma_data[days] = dict()
        for i in range(len(self.data)):
            ma = 0
            added = 0
            for j in range(i - days, i):
                if j >= 0:
                    added += 1
                    ma += self.data[j]
            ma /= added
            self.ma_data[days][self.times[i]] = ma

    def get_ma_value(self, time, days: int = 7) -> float:
        """
        Returns the backwards moving average at the given time.
        :param time: time to evaluate the average
        :param days: number of days to average
        :return:
        """
        if not days in self.ma_data.keys():
            self._make_ma(days)
        try:
            return self.ma_data[days][time]
        except:
            return self.default

    def get_ma_values(self, days: int = 7) -> list[float]:
        """
        Returns the backwards moving average as a list.
        Note that the resulting list will be "days-1" shorter than the list returned by "get_times"
        :param days: number of days to average
        :return:
        """
        if not days in self.ma_data.keys():
            self._make_ma(days)
        return [self.ma_data[days][t] for t in self.times]

    def append_value(self, time, value: float) -> None:
        """
        Appends a value to the time series
        :param time:
        :param value:
        :return:
        """
        self.data[time] = value
        self.times.append(time)
        self.ma_data = dict()

    def add_value(self, time, value: float) -> None:
        """
        Either adds a value to a given value of time series at the given time, or creates it.
        Note that this different from "append_value"!
        :param time:
        :param value:
        :return:
        """
        try:
            self.data[time] += value
        except:
            self.append_value(time, value)
        self.ma_data = dict()

    def multiply_value(self, time, value: float) -> None:
        """
        Either multiplies a value to given value of time series at the given time, or creates it.
        :param time:
        :param value:
        :return:
        """
        try:
            self.data[time] *= value
        except:
            self.append_value(time, value)
        self.ma_data = dict()


def parse_date(datestring: str) -> dt.date:
    """
    convencience method to parse date strings with different typical formats
    :param datestring:
    :return:
    """
    try:
        return dt.datetime.strptime(datestring[:10], '%Y-%m-%d').date()
    except:
        try:
            return dt.datetime.strptime(datestring[:10], '%d.%m.%Y').date()
        except:
            print(datestring)
            raise ()
