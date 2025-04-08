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
import datetime as dt
from config import Config

class VaccinationParameters:
    def __init__(self,config:Config) -> None:
        """
        Interface class between vaccination data and simulation.
        :param config: config instance of the simulation
        """
        self.config = config
        self.vaccinations = dict()
        self.vaccinationsFed = dict()
        #parse CSV file
        file = config.filenameVaccdata
        with open(file,'r') as f:
          r = csv.reader(f,delimiter=';')
          next(r)
          for line in r:
              date = dt.datetime.strptime(line[0],"%Y-%m-%d")
              fed = line[1]
              shotNo = int(line[2])
              count = int(line[3])
              try:
                  self.vaccinations[date][shotNo]+=count
              except:
                  try:
                      self.vaccinations[date][shotNo] = count
                  except:
                      self.vaccinations[date] = dict()
                      self.vaccinations[date][shotNo] = count
              if fed not in self.vaccinationsFed.keys():
                  self.vaccinationsFed[fed]=dict()
              try:
                  self.vaccinationsFed[fed][date][shotNo] += count
              except:
                  try:
                      self.vaccinationsFed[fed][date][shotNo] = count
                  except:
                      self.vaccinationsFed[fed][date] = dict()
                      self.vaccinationsFed[fed][date][shotNo] = count

    def get(self,date:dt.datetime,shotNo:int,fed=None) -> int:
        """
        Returns the number of 'shotNo'-vaccine shots for the given date in the given federalstate
        :param date: date of the vaccination
        :param shotNo: shot-number (1,2,3)
        :param fed: optional, federalstate
        :return: number of shots for the date
        """
        if fed==None:
            try:
                return self.vaccinations[date][shotNo]
            except:
                return 0
        else:
            try:
                return self.vaccinationsFed[fed][date][shotNo]
            except:
                return 0
