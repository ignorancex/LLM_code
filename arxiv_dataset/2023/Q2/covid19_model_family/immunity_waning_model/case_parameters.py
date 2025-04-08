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
import datetime as dt

from detection_parameters import DetectionParameters

class CaseParameters:
    def __init__(self,config:Config) -> None:
        """
        Class for managing the data interface between epidemiological case data and simulation.
        :param config: config file of the simulation
        """
        self.config=config
        self.cases = dict()
        self.casesFed = dict()
        file = config.filenameEpidata
        detection_parameters = DetectionParameters(config)

        #parse epidemiological data
        with open(file, 'r') as f:
            r = csv.reader(f, delimiter=';')
            next(r)
            for line in r:
                date = dt.datetime.strptime(line[0], "%Y-%m-%d")
                fed = line[1]
                count = int(line[2])
                try:
                    self.cases[date][True] += count
                    self.cases[date][False] += count/detection_parameters.get_detection_probability(date)*(1-detection_parameters.get_detection_probability(date))
                except:
                    try:
                        self.cases[date][True] = count
                        self.cases[date][False] = count / detection_parameters.get_detection_probability(date) * (
                                    1 - detection_parameters.get_detection_probability(date))
                    except:
                        self.cases[date] = dict()
                        self.cases[date][True] = count
                        self.cases[date][False] = count / detection_parameters.get_detection_probability(date) * (
                                    1 - detection_parameters.get_detection_probability(date))
                if fed not in self.casesFed.keys():
                    self.casesFed[fed]=dict()
                try:
                    self.casesFed[fed][date][True] += count
                    self.casesFed[fed][date][False] += count / detection_parameters.get_detection_probability(date) * (
                            1 - detection_parameters.get_detection_probability(date))
                except:
                    try:
                        self.casesFed[fed][date][True] = count
                        self.casesFed[fed][date][False] = count / detection_parameters.get_detection_probability(date) * (
                                1 - detection_parameters.get_detection_probability(date))
                    except:
                        self.casesFed[fed][date] = dict()
                        self.casesFed[fed][date][True] = count
                        self.casesFed[fed][date][False] = count / detection_parameters.get_detection_probability(date) * (
                                1 - detection_parameters.get_detection_probability(date))
        #extrapolate timeseries a little - otherwise undetected cases stop increase/decrease "too early"
        date = max(self.cases.keys())
        count = sum([self.cases[date-dt.timedelta(k)][True] for k in range(7)])/7
        for k in range(1,8):
            self.cases[date + dt.timedelta(k)] = dict()
            self.cases[date + dt.timedelta(k)][True] = count
            self.cases[date + dt.timedelta(k)][False] = count / detection_parameters.get_detection_probability(date) * (
                    1 - detection_parameters.get_detection_probability(date))

    def get(self,date:dt.datetime,detected:bool,fed=None) -> float:
        """
        Returns a smoothed estimate for number of detected or undetected cases for the given confirmation-date, i.e. the date they are registered in the surveillance system. For undetected cases, the method returns an estimate. Note that those undetected cases would have become registered at the given date, if they would have made a test.
        :param date: date to get data for
        :param detected: true returns confirmed cases, false return extimate for undetected cases
        :param fed: Austrian federalstate (e.g. AT-9 for Vienna)
        :return: number of new cases for given date, region and detection status
        """
        x = 0
        #smooth weekly bias
        for k in self.config.detDelay:
            x+=self._get(date+dt.timedelta(k),detected,fed)
        return x/len(self.config.detDelay)

    def _get(self,date:dt.datetime,detected:bool,fed=None) -> int:
        """
        Called seven times by "get" to smooth the numbers. Returns an estimate for number of detected or undetected cases for the given confirmation-date, i.e. the date they are registered in the surveillance system. For undetected cases, the method returns an estimate. Note that those undetected cases would have become registered at the given date, if they would have made a test.
        :param date: date to get data for
        :param detected: true returns confirmed cases, false return extimate for undetected cases
        :param fed: Austrian federalstate (e.g. AT-9 for Vienna)
        :return: number of new cases for given date, region and detection status
        """
        if fed==None:
            try:
                return self.cases[date][detected]
            except:
                return 0
        else:
            try:
                return self.casesFed[fed][date][detected]
            except:
                return 0