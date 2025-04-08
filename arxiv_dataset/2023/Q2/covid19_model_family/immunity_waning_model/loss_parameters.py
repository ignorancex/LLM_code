"""
Copyright (C) 2023 dwh GmbH - All Rights Reserved
You may use, distribute and modify this code under the 
terms of the MIT license.

You should have received a copy of the MIT license with
this file. If not, please write to: 
martin.bicher@dwh.at or visit 
https://github.com/dwhGmbH/covid19_model_family/blob/main/LICENSE.txt
"""


from loss_sampler import LossSampler
from config import Config

class LossParameters:
    def __init__(self,config:Config):
        """
        Class to manage sampling of waning dates.
        Key of this class are the five parameters 'samplefunVacc1-samplefunRecOther' which can be called with one parameter (typically the mean value of the distribution) and randomly generate a new immunity-loss duration.
        Dependent of the corresponding string-fields in the config, different distributions are used.
        :param config: config instance of the simulation
        """
        self.config=config
        self.samplers = dict()
        for key1, value1 in self.config.observables.items():
            self.samplers[key1]=dict()
            for key2, value2 in value1.items():
                ls = LossSampler(value2['distribution'],value2['mean'])
                self.samplers[key1][key2]=LossSampler(value2['distribution'],value2['mean'])

    def sample_loss(self, cause:str, targets:list[str]) -> list[int]:
        """
        Samples a waning duration for immunization event against a list of targets and a given immunization cause
        :param cause: typically either VACC1,2,.. or ALPHA,DELTA,...
        :param targets: typically either ALPHA,DELTA,... or an other given observable
        :return: waning duration in days
        """
        out = list()
        try:
            val =  self.samplers[targets[0]][cause].sample()
            mean0 = self.samplers[targets[0]][cause].mean
        except:
            val = self.samplers[targets[0]]['DEFAULT'].sample()
            mean0 = self.samplers[targets[0]]['DEFAULT'].mean
        out.append(val)
        for target in targets[1:]:
            try:
                mean1 = self.samplers[target][cause].mean
            except:
                mean1 = self.samplers[target]['DEFAULT'].mean
            out.append(int(val*mean1/mean0))
        return out