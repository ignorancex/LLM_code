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
from typing import Tuple

import numpy as np

from config import Config
import datetime as dt

class ResultExporter:
    def __init__(self) -> None:
        """
        Class to export the simulation result
        """
        self.timestamp = dt.datetime.now().strftime('%Y%m%d_%H%M%S')

    def export_to_csv(self,config:Config,result:dict) -> str:
        """
        Exports the simulation result in doct format to a csv file
        :param config: config instance of the simulation
        :param result: simulation result as dict object
        :return:
        """
        outputfile = os.path.join(config.get_result_folder(),config.get_csv_filename())
        X = list()
        header = list(result.keys())
        header.sort()
        header.remove('time')
        header.insert(0,'time')
        X.append(header)
        for i in range(len(result['time'])):
            line = [round(result[k][i],0) if k!='time' else result[k][i] for k in header]
            X.append(line)

        #  if there is an existing csv file in the experiment folder, dont override but extend it
        if os.path.isfile(outputfile):
            with open(outputfile,'r') as f:
                r = csv.reader(f,delimiter=';')
                Y = [x for x in r]
                for i,h in enumerate(Y[0]):
                    if h not in header:
                        for k in range(len(X)):
                            X[k].append(Y[k][i])
        #writing the csv file
        with open(outputfile,'w',newline='') as f:
            w = csv.writer(f,delimiter=';')
            w.writerows(X)
        return outputfile


    def load_from_csv(self, filename: str) -> Tuple[dict, list]:
        """
        Load the result from the csv file
        :param filename: csv-filename
        :return: dict with loaded results and list of variants that occur in the result
        """
        with open(filename, 'r') as f:
            r = csv.reader(f, delimiter=';')
            header = next(r)
            result = {k: list() for k in header}
            result['time'] = list()
            for i, line in enumerate(r):
                for v, k in zip(line, header):
                    if k == 'time' or k == '':
                        result["time"].append(dt.datetime.strptime(v[:10], '%Y-%m-%d'))
                    else:
                        result[k].append(float(v))

        variants = [x[16:] for x in result.keys() if x.startswith('active detected ')]
        return result, variants
