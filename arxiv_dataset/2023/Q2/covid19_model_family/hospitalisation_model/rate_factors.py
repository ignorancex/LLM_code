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

from config import Config
from utils import TimeSeries, parse_date


class RateFactors:
    """
    Class for interaction between rate number file and simulation
    """

    def __init__(self, config: Config) -> None:
        """
        Class for interaction between rate number file and simulation
        :param config: config instance
        """
        if config.get_filename_case_numbers() == None:
            self.rateFactors = {i: None for i in range(config.get_column_number())}
        else:
            self.rateFactors = {i: TimeSeries(list(), list(), default=1.0) for i in range(config.get_column_number())}
            with open(config.get_filename_rate_factors()) as f:
                r = csv.reader(f, delimiter=';')
                hdr = next(r)
                inds = [hdr.index(x) for x in config.get_columns_rate_factors()]
                for line in r:
                    day = parse_date(line[0])
                    for i in range(len(inds)):
                        self.rateFactors[i].append_value(day, float(line[inds[i]]))

    def get_factors(self, index: int) -> TimeSeries:
        """
        :param index: index of scenario=column
        :return: TimeSeries object of the rate numbers for the given index
        """
        return self.rateFactors[index]
