"""
Copyright (C) 2023 Martin Bicher - All Rights Reserved
You may use, distribute and modify this code under the 
terms of the MIT license.

You should have received a copy of the MIT license with
this file. If not, please write to: 
martin.bicher@tuwien.ac.at or visit 
https://github.com/dwhGmbH/covid19_model_family/LICENSE.txt
"""

from config import Config
import numpy as np
import csv
import datetime as dt


class VariantParameters:
    def __init__(self, config:Config=None) -> None:
        """
        Interface between variant information and simulation. The variant .csv file is specified as follows:
        Each column of the file corresponds to a variant. The value in each line corresponds to the ratio of daily cases
        Instance is also used to randomly sample a variant for a given case.
        :param config: config instance of the simulation
        """
        self.variants = list()
        self.ratios = dict()
        # parse variant csv file
        if config == None:
            filename = 'variant_data.csv'
        else:
            filename = config.filenameVariantdata
        with open(filename, 'r') as f:
            r = csv.reader(f, delimiter=';')
            hdr = next(r)
            self.variants = hdr[1:]
            for line in r:
                date = dt.datetime.strptime(line[0], '%Y-%m-%d').date()
                ratios = [float(x) for x in line[1:]]
                sm = sum(ratios)  # just to make sure...
                ratios = [x / sm for x in ratios]
                self.ratios[date] = ratios
        self.maxtime = max(self.ratios.keys())
        self.mintime = max(self.ratios.keys())

    def get_variants(self) -> list:
        """
        Returns the internally used list of variants
        :return: list of variant names
        """
        return self.variants

    def get_variant_ratios(self, times: list) -> list:
        """
        Evaluates the variant splits for a given time list. Utilizes the :func:get_variant_ratio function for each date.
        :param times: list of date values
        :return: list of lists with length of the input timeline. The inner lists sum up to one and correspond to the outcome of :func:get_variants
        """
        ratios = list()
        for t in times:
            ratios.append(self.get_variant_ratio(t.date()))
        return ratios

    def get_variant_ratio(self, time: dt.date) -> list:
        """
        Evaluates the variant splits for a given time. Utilizes the :func:_get_value function for each date.
        :param times: list of date values
        :return: the list sums up to one and corresponds to the outcome of :func:get_variants
        """
        try:
            return self.ratios[time]
        except:
            if time > self.maxtime:
                return self.ratios[self.maxtime]
            elif time < self.mintime:
                return self.ratios[self.mintime]
            else:
                raise ValueError

    def sample_variant(self, time: dt.date) -> str:
        """
        Samples a random variant for the given date
        :param time: current date
        :return: variant name as string
        """
        lst = self.get_variant_ratio(time)
        return np.random.choice(self.variants, p=lst)