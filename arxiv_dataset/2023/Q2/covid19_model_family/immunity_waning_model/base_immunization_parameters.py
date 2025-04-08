"""
Copyright (C) 2023 Martin Bicher - All Rights Reserved
You may use, distribute and modify this code under the 
terms of the MIT license.

You should have received a copy of the MIT license with
this file. If not, please write to: 
martin.bicher@tuwien.ac.at or visit 
https://github.com/dwhGmbH/covid19_model_family/LICENSE.txt
"""

import numpy as np
from config import Config

class BaseImmunizationParameters:
    def __init__(self,config:Config):
        """
        Class to manage sampling of base immunity
        :param config: config instance of the simulation
        """
        self.config=config
        self.baseValues = dict()
        for key1, value1 in self.config.observables.items():
            self.baseValues[key1]=dict()
            for key2, value2 in value1.items():
                self.baseValues[key1][key2]= value2['base']

    def sample_base_immunity(self,cause:str,target:str)->bool:
        """
        Samples a base immunity for immunization event against a certain target and a given immunization cause
        :param cause: typically either VACC1,2,.. or ALPHA,DELTA,...
        :param target: typically either ALPHA,DELTA,... or an other given observable
        :return: whether the immunization event was successful
        """
        rand = np.random.random()
        try:
            return rand<self.baseValues[target][cause]
        except:
            return rand<self.baseValues[target]['DEFAULT']

    def sample_base_immunity_all(self,cause:str,targets:list[str])->list[bool]:
        """
        Samples a base immunity for immunization event against a certain target and a given immunization cause
        :param cause: typically either VACC1,2,.. or ALPHA,DELTA,...
        :param targets: typically either ALPHA,DELTA,... or an other given observable
        :return: whether the immunization event was successful
        """
        rand = np.random.random()
        out = list()
        for target in targets:
            try:
                val = rand<self.baseValues[target][cause]
            except:
                val = rand<self.baseValues[target]['DEFAULT']
            out.append(val)
        return out