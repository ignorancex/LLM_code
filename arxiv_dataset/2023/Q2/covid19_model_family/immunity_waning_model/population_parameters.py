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
from utils import *

class PopulationParameters:
    def __init__(self,config:Config) -> None:
        """
        Class for managing the data interface between population data and simulation.
        :param config: config file of the simulation
        """
        self.config=config
        self.population = dict()
        file = config.filenamePopulationdata

        #parse data
        with open(file, 'r') as f:
            r = csv.reader(f, delimiter=';')
            next(r)
            for line in r:
                fed = line[0]
                count = int(line[1])
                self.population[fed]=count
        self.federalstates = list(self.population.keys())

    def get_federalstates(self) ->  list:
        return self.federalstates

    def get_population(self,fed:str=None) -> float:
        """
        Returns a the number of inhabitants of a given federalstate. If None is passed, then the sum is returned
        :param fed: regionID or None for the whole counry
        """
        if fed == None:
            return sum(self.population.values())
        else:
            return self.population[fed]